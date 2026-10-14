#include "core/attention_ops.cuh"
#include "core/tensor_ops.cuh"
#include "../ordinary_norm/buffers.cuh"
using namespace ferrule::cuda::core;
#include "candidate.cuh"
__global__ void convert(const float *x,uint16_t*y,uint64_t n){uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)y[i]=bf16_rne(x[i]);}
int main(int argc,char**argv){try{
  require(argc==8,"dir rows heads kvheads dim context pages");std::string d=argv[1];
  uint32_t rows=std::stoul(argv[2]),heads=std::stoul(argv[3]),kh=std::stoul(argv[4]),dim=std::stoul(argv[5]),ctx=std::stoul(argv[6]),pages=std::stoul(argv[7]);
  uint64_t n=uint64_t(rows)*heads*dim,cache_n=uint64_t(pages+1)*16*kh*dim;
  Buffer<float>q(n),out(n),tracebuf(2*rows*heads*ctx);Buffer<uint16_t>k(cache_n),v(cache_n),gate(n),raw(n),gated(n);
  Buffer<int32_t>slots(pages),offsets(2),seq(rows),pos(rows),len(rows),status(1);
  q.read(d+"/query.f32");k.read(d+"/cache-key.bf16");v.read(d+"/cache-value.bf16");gate.read(d+"/gate.bf16");slots.read(d+"/slots.i32");offsets.read(d+"/offsets.i32");seq.read(d+"/sequences.i32");pos.read(d+"/positions.i32");len.read(d+"/lens.i32");ck(cudaMemset(status.p,0,4));
  FerruleCoreTransformerArgs a{};a.kind=FERRULE_CORE_TRANSFORMER_PAGED_BF16_CAUSAL_GQA;a.flags=1;a.rows=rows;a.sequences=1;a.q_heads=heads;a.kv_heads=kh;a.head_dim=dim;a.page_tokens=16;a.layer_count=1;a.softmax_scale=float(1./sqrt(double(dim)));
  a.query_f32=q.addr();a.query_bytes=n*4;a.query_row_stride_bytes=heads*dim*4;a.query_head_stride_bytes=dim*4;
  a.key_cache_bf16=k.addr();a.value_cache_bf16=v.addr();a.key_cache_bytes=a.value_cache_bytes=cache_n*2;a.key_slot_stride_bytes=a.value_slot_stride_bytes=16*kh*dim*2;a.key_layer_stride_bytes=a.value_layer_stride_bytes=16*kh*dim*2;a.key_token_stride_bytes=a.value_token_stride_bytes=kh*dim*2;a.key_head_stride_bytes=a.value_head_stride_bytes=dim*2;
  a.block_slots_i32=slots.addr();a.block_slots_count=pages;a.block_offsets_i32=offsets.addr();a.block_offsets_count=2;
  a.row_sequence_ids_i32=seq.addr();a.row_sequence_ids_count=rows;a.row_positions_i32=pos.addr();a.row_positions_count=rows;a.row_kv_lens_i32=len.addr();a.row_kv_lens_count=rows;
  a.output_f32=out.addr();a.output_bytes=n*4;a.output_row_stride_bytes=heads*dim*4;a.output_head_stride_bytes=dim*4;a.status_i32=status.addr();a.status_count=1;
  require(valid_transformer(&a),"native descriptor");eager_candidate<<<rows*heads,128>>>(a,tracebuf.p);ck(cudaGetLastError());convert<<<(n+255)/256,256>>>(out.p,raw.p,n);ck(cudaGetLastError());raw.write(d+"/candidate-raw.bf16");tracebuf.write(d+"/candidate-stages.f32");out.write(d+"/candidate-pv.f32");
  FerruleCoreDataArgs b{};b.kind=FERRULE_CORE_DATA_SIGMOID_GATE_BF16;b.count=n;b.width=heads*dim;b.rows=rows;b.input0=raw.addr();b.input1=gate.addr();b.output0=gated.addr();data_kernel<<<(n+255)/256,256>>>(b);ck(cudaGetLastError());gated.write(d+"/candidate-gated.bf16");require(status.host()[0]==0,"device metadata");q.immutable();k.immutable();v.immutable();gate.immutable();std::cout<<"candidate output and gate; all guards pass\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
