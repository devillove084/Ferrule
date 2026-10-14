#include "core/attention_ops.cuh"
#include "core/tensor_ops.cuh"
#include "../ordinary_norm/buffers.cuh"
using namespace ferrule::cuda::core;
#include "candidate.cuh"
__global__ void trace(FerruleCoreTransformerArgs a, float *stages, uint16_t *raw,
                      uint64_t gate, uint16_t *sigmoid) {
  const uint32_t h=threadIdx.x; if(h>=a.q_heads)return;
  const uint32_t n=11,dim=a.head_dim,kh=h/(a.q_heads/a.kv_heads);
  const uint64_t q=uint64_t(h)*dim*4;
  float scores[11],maximum=-CUDART_INF_F;
  for(uint32_t t=0;t<n;++t){
    float dot=0;
    for(uint32_t j=0;j<dim;++j){uint64_t off;transformer_cache_offset(a,0,t,kh,j,false,&off);dot+=const_pointer<float>(a.query_f32+q)[j]*bf16_value(const_pointer<uint16_t>(a.key_cache_bf16+off)[0]);}
    stages[h*n+t]=dot;stages[16*n+h*n+t]=bf16_round(dot);
    transformer_gqa_score<uint16_t>(a,0,t,kh,q,&scores[t]);
    stages[2*16*n+h*n+t]=scores[t];maximum=fmaxf(maximum,scores[t]);
  }
  float denominator=0;
  for(uint32_t t=0;t<n;++t)denominator+=expf(scores[t]-maximum);
  for(uint32_t t=0;t<n;++t){float p=expf(scores[t]-maximum)/denominator;stages[3*16*n+h*n+t]=p;stages[4*16*n+h*n+t]=bf16_round(p);}
  for(uint32_t j=0;j<dim;++j){float out=0;
    for(uint32_t t=0;t<n;++t){uint64_t off;transformer_cache_offset(a,0,t,kh,j,true,&off);out+=stages[4*16*n+h*n+t]*bf16_value(const_pointer<uint16_t>(a.value_cache_bf16+off)[0]);}
    raw[h*dim+j]=bf16_rne(out);
    float g=bf16_value(const_pointer<uint16_t>(gate)[h*dim+j]);sigmoid[h*dim+j]=bf16_rne(1.f/(1.f+expf(-g)));
  }
}
__global__ void cast(const float *x,uint16_t *y){uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<4096)y[i]=bf16_rne(x[i]);}
int main(int argc,char **argv){try{
  require(argc==2,"prepared directory");std::string d=argv[1];
  cudaDeviceProp prop;int runtime;ck(cudaGetDeviceProperties(&prop,0));ck(cudaRuntimeGetVersion(&runtime));require(runtime==12040&&prop.major==8&&prop.minor==6,"CUDA12.4 sm86");
  Buffer<float>q(4096),out(4096),stages(5*16*11);Buffer<uint16_t>k(16*2*256),v(k.n),gate(4096),raw(4096),traced(4096),sigmoid(4096),gated(4096),hfraw(4096);
  Buffer<int32_t>slots(1),offsets(2),sequences(1),positions(1),lens(1),status(1);
  q.read(d+"/query.f32");k.read(d+"/cache-key.bf16");v.read(d+"/cache-value.bf16");gate.read(d+"/gate.bf16");hfraw.read(d+"/expected_ungated.bf16");
  slots.read(d+"/slots.i32");offsets.read(d+"/offsets.i32");sequences.read(d+"/sequences.i32");positions.read(d+"/positions.i32");lens.read(d+"/lens.i32");ck(cudaMemset(status.p,0,4));
  FerruleCoreTransformerArgs a{};a.kind=FERRULE_CORE_TRANSFORMER_PAGED_BF16_CAUSAL_GQA;a.flags=1;a.rows=a.sequences=1;a.q_heads=16;a.kv_heads=2;a.head_dim=256;a.page_tokens=16;a.layer_count=1;a.softmax_scale=.0625f;
  a.query_f32=q.addr();a.query_bytes=4096*4;a.query_row_stride_bytes=4096*4;a.query_head_stride_bytes=256*4;
  a.key_cache_bf16=k.addr();a.value_cache_bf16=v.addr();a.key_cache_bytes=a.value_cache_bytes=k.n*2;
  a.key_slot_stride_bytes=a.value_slot_stride_bytes=k.n*2;a.key_layer_stride_bytes=a.value_layer_stride_bytes=k.n*2;
  a.key_token_stride_bytes=a.value_token_stride_bytes=2*256*2;a.key_head_stride_bytes=a.value_head_stride_bytes=256*2;
  a.block_slots_i32=slots.addr();a.block_slots_count=1;a.block_offsets_i32=offsets.addr();a.block_offsets_count=2;
  a.row_sequence_ids_i32=sequences.addr();a.row_sequence_ids_count=1;a.row_positions_i32=positions.addr();a.row_positions_count=1;a.row_kv_lens_i32=lens.addr();a.row_kv_lens_count=1;
  a.output_f32=out.addr();a.output_bytes=4096*4;a.output_row_stride_bytes=4096*4;a.output_head_stride_bytes=256*4;a.status_i32=status.addr();a.status_count=1;
  require(valid_transformer(&a),"actual layout validation");transformer_causal_gqa_kernel<uint16_t><<<16,256>>>(a);ck(cudaGetLastError());cast<<<16,256>>>(out.p,raw.p);ck(cudaGetLastError());raw.write(d+"/native-raw.bf16");out.write(d+"/native-pv-f32.f32");
  trace<<<1,32>>>(a,stages.p,traced.p,gate.addr(),sigmoid.p);ck(cudaGetLastError());require(raw.host()==traced.host(),"trace raw output differs");stages.write(d+"/native-stages.f32");sigmoid.write(d+"/native-sigmoid.bf16");
  FerruleCoreDataArgs b{};b.kind=FERRULE_CORE_DATA_SIGMOID_GATE_BF16;b.count=b.width=4096;b.rows=1;b.input0=raw.addr();b.input1=gate.addr();b.output0=gated.addr();data_kernel<<<16,256>>>(b);ck(cudaGetLastError());gated.write(d+"/native-gated.bf16");
  b.input0=hfraw.addr();data_kernel<<<16,256>>>(b);ck(cudaGetLastError());gated.write(d+"/native-gated-hfraw.bf16");
  eager_candidate<<<16,128>>>(a,stages.p);ck(cudaGetLastError());cast<<<16,256>>>(out.p,raw.p);ck(cudaGetLastError());raw.write(d+"/candidate-raw.bf16");stages.write(d+"/candidate-stages.f32");b.input0=raw.addr();data_kernel<<<16,256>>>(b);ck(cudaGetLastError());gated.write(d+"/candidate-gated.bf16");
  require(status.host()[0]==0,"metadata status");q.immutable();k.immutable();v.immutable();gate.immutable();std::cout<<"actual paged eager replay/traces exact; gate isolated; input/guards pass\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
