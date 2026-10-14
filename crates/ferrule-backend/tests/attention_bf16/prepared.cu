#include "core/attention_ops.cuh"
#include "core/tensor_ops.cuh"
#include "../ordinary_norm/buffers.cuh"
using namespace ferrule::cuda::core;
static std::string evidence;
static bool observing=true;
__device__ float *softmax_trace;
__device__ uint64_t softmax_extent;
static void observe(const char*name,const uint16_t*p,uint64_t n){
  if(!observing)return;
  std::vector<uint16_t> host(n);ck(cudaMemcpy(host.data(),p,n*2,cudaMemcpyDeviceToHost));
  std::ofstream f(evidence+"/native-"+name+".bf16",std::ios::binary);
  require(bool(f.write(reinterpret_cast<const char*>(host.data()),n*2)),"stage write");
}
#define FERRULE_ATTENTION_OBSERVE(name,ptr,n) observe(name,ptr,n)
#define FERRULE_ATTENTION_SOFTMAX_TRACE(stage,index,value) (softmax_trace[uint64_t(stage)*softmax_extent+(index)]=(value))
#include "core/attention_bf16.cuh"
__global__ void convert(const float*x,uint16_t*y,uint64_t n){uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)y[i]=bf16_rne(x[i]);}
struct Owner {uint64_t h=0;~Owner(){if(h)ferrule_bf16_attention_destroy(h);}};
int main(int argc,char**argv){try{
  if(argc==2&&std::string(argv[1])=="--cpu-contract"){
    using namespace ferrule::cuda::core::eager_attention;
    FerruleBf16AttentionPlan p{};p.heads=16;p.kv_heads=2;p.dim=256;p.max_rows=5;p.max_tokens=129;
    uint64_t sizes[]={40960,1056768,1056768,20640,20640,40960,32ull<<20};
    for(int i=0;i<7;++i){p.bytes=(p.bytes+255)&~255ull;p.offsets[i]=p.bytes;p.bytes+=sizes[i];}
    require(plan_valid(p),"native plan");require(p.bytes==35791360,"Rust/native budget");
    require(memory_valid(p,256,p.bytes),"aligned workspace");require(!memory_valid(p,258,p.bytes),"workspace alignment");
    require(!memory_valid(p,256,p.bytes-1),"short workspace");require(!memory_valid(p,UINT64_MAX-255,p.bytes),"pointer overflow");
    auto bad=p;bad.heads=UINT32_MAX;require(!plan_valid(bad),"head overflow");
    bad=p;bad.offsets[4]=bad.offsets[3];require(!plan_valid(bad),"panel overlap");
    bad=p;bad.max_tokens=1025;require(!plan_valid(bad),"softmax bound");
    std::cout<<"8 native ABI/extent/budget rejection checks; no CUDA calls\n";return 0;
  }
  require(argc==9,"dir rows heads kvheads dim context pages append");std::string d=argv[1];evidence=d;
  uint32_t rows=std::stoul(argv[2]),heads=std::stoul(argv[3]),kh=std::stoul(argv[4]),dim=std::stoul(argv[5]),ctx=std::stoul(argv[6]),pages=std::stoul(argv[7]);bool app=std::stoul(argv[8]);
  uint64_t n=uint64_t(rows)*heads*dim,cache_n=uint64_t(pages+1)*16*kh*dim,extent=uint64_t(rows)*heads*ctx;
  Buffer<float>q(n),out(n),trace(3*extent);Buffer<uint16_t>k(cache_n),v(cache_n),gate(n),raw(n),gated(n),ak(uint64_t(rows)*kh*dim),av(uint64_t(rows)*kh*dim);
  Buffer<int32_t>slots(pages),offsets(2),seq(rows),pos(rows),len(rows),status(1);
  q.read(d+"/query.f32");k.read(d+"/cache-key.bf16");v.read(d+"/cache-value.bf16");gate.read(d+"/gate.bf16");slots.read(d+"/slots.i32");offsets.read(d+"/offsets.i32");seq.read(d+"/sequences.i32");pos.read(d+"/positions.i32");len.read(d+"/lens.i32");
  ak.read(d+"/append-key.bf16");av.read(d+"/append-value.bf16");
  if(app){for(int32_t position:pos.original){uint64_t i=uint64_t(slots.original[position/16])*16*kh*dim+uint64_t(position%16)*kh*dim;ck(cudaMemset(k.p+i,0xff,kh*dim*2));ck(cudaMemset(v.p+i,0xff,kh*dim*2));}}
  FerruleCoreTransformerArgs a{};a.kind=app?FERRULE_CORE_TRANSFORMER_PAGED_BF16_APPEND_CAUSAL_GQA:FERRULE_CORE_TRANSFORMER_PAGED_BF16_CAUSAL_GQA;a.flags=1;a.rows=rows;a.sequences=1;a.q_heads=heads;a.kv_heads=kh;a.head_dim=dim;a.page_tokens=16;a.layer_count=1;a.softmax_scale=float(1./sqrt(double(dim)));
  a.query_f32=q.addr();a.query_bytes=n*4;a.query_row_stride_bytes=heads*dim*4;a.query_head_stride_bytes=dim*4;
  a.append_key_bf16=ak.addr();a.append_value_bf16=av.addr();a.append_key_bytes=a.append_value_bytes=uint64_t(rows)*kh*dim*2;
  a.append_key_row_stride_bytes=a.append_value_row_stride_bytes=kh*dim*2;a.append_key_head_stride_bytes=a.append_value_head_stride_bytes=dim*2;
  a.key_cache_bf16=k.addr();a.value_cache_bf16=v.addr();a.key_cache_bytes=a.value_cache_bytes=cache_n*2;a.key_slot_stride_bytes=a.value_slot_stride_bytes=16*kh*dim*2;a.key_layer_stride_bytes=a.value_layer_stride_bytes=16*kh*dim*2;a.key_token_stride_bytes=a.value_token_stride_bytes=kh*dim*2;a.key_head_stride_bytes=a.value_head_stride_bytes=dim*2;
  a.block_slots_i32=slots.addr();a.block_slots_count=pages;a.block_offsets_i32=offsets.addr();a.block_offsets_count=2;
  a.row_sequence_ids_i32=seq.addr();a.row_sequence_ids_count=rows;a.row_positions_i32=pos.addr();a.row_positions_count=rows;a.row_kv_lens_i32=len.addr();a.row_kv_lens_count=rows;
  a.output_f32=out.addr();a.output_bytes=n*4;a.output_row_stride_bytes=heads*dim*4;a.output_head_stride_bytes=dim*4;a.status_i32=status.addr();a.status_count=1;
  FerruleBf16AttentionPlan p{};p.heads=heads;p.kv_heads=kh;p.dim=dim;p.max_rows=rows;p.max_tokens=ctx;
  uint64_t sizes[]={n*2,uint64_t(heads)*ctx*dim*2,uint64_t(heads)*ctx*dim*2,extent*2,extent*2,n*2,32ull<<20};
  for(int i=0;i<7;++i){p.bytes=(p.bytes+255)&~255ull;p.offsets[i]=p.bytes;p.bytes+=sizes[i];}
  Buffer<uint4> workspace((p.bytes+15)/16);Owner owner;uint64_t growth=0;int version=0;
  FerruleBf16AttentionGroup group{0,rows,0,ctx};
  require(ferrule_bf16_attention_create(&p,&group,1,0,workspace.addr(),workspace.n*16,&owner.h,&growth,&version)==0,"native preparation");
  ck(cudaMemcpyToSymbol(softmax_trace,&trace.p,sizeof(trace.p)));ck(cudaMemcpyToSymbol(softmax_extent,&extent,sizeof(extent)));
  require(ferrule_bf16_attention_begin(0,status.addr())==0,"explicit invocation clear");
  for(int repeat=0;repeat<2;++repeat){
    ck(cudaMemset(workspace.p,repeat?0xa5:0x5a,p.offsets[6]));
    require(ferrule_bf16_attention_launch(&a,&p,&group,1,workspace.addr(),workspace.n*16,owner.h)==0,"native launch");
    require(status.host()[0]==0,"device metadata");
    convert<<<(n+255)/256,256>>>(out.p,raw.p,n);ck(cudaGetLastError());raw.write(d+"/native-raw-"+std::to_string(repeat)+".bf16");
    trace.write(d+"/native-softmax-"+std::to_string(repeat)+".f32");
    FerruleCoreDataArgs b{};b.kind=FERRULE_CORE_DATA_SIGMOID_GATE_BF16;b.count=n;b.width=heads*dim;b.rows=rows;b.input0=raw.addr();b.input1=gate.addr();b.output0=gated.addr();data_kernel<<<(n+255)/256,256>>>(b);ck(cudaGetLastError());gated.write(d+"/native-gated-"+std::to_string(repeat)+".bf16");
  }
  auto bad=p;bad.offsets[4]=bad.offsets[3];require(ferrule_bf16_attention_launch(&a,&bad,&group,1,workspace.addr(),workspace.n*16,owner.h)==1,"workspace overlap reject");
  auto alias=a;alias.output_f32=a.query_f32;require(ferrule_bf16_attention_launch(&alias,&p,&group,1,workspace.addr(),workspace.n*16,owner.h)==1,"alias reject");
  require(ferrule_bf16_attention_launch(&a,&p,&group,1,workspace.addr(),p.bytes-1,owner.h)==1,"short workspace reject");
  observing=false;
  auto before=out.host();int32_t invalid=-1;ck(cudaMemcpy(pos.p,&invalid,4,cudaMemcpyHostToDevice));
  require(ferrule_bf16_attention_launch(&a,&p,&group,1,workspace.addr(),workspace.n*16,owner.h)==0,"invalid metadata submission");
  require(status.host()[0]!=0,"invalid metadata status");require(before==out.host(),"invalid metadata must not publish output");
  ck(cudaMemcpy(pos.p,pos.original.data(),rows*4,cudaMemcpyHostToDevice));
  q.immutable();k.immutable();v.immutable();gate.immutable();ak.immutable();av.immutable();slots.immutable();offsets.immutable();seq.immutable();pos.immutable();len.immutable();workspace.host();
  std::cout<<"prepared full stages, dirty reuse, append="<<app<<", metadata/alias/extent guards; cublas="<<version<<", growth="<<growth<<'\n';
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
