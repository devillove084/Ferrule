#pragma once
#include "core/attention_ops.cuh"
#include "core/attention_bf16_abi.h"
#include <cublas_v2.h>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <initializer_list>
#ifndef FERRULE_ATTENTION_OBSERVE
#define FERRULE_ATTENTION_OBSERVE(name,ptr,n) ((void)(n))
#endif
#ifndef FERRULE_ATTENTION_SOFTMAX_TRACE
#define FERRULE_ATTENTION_SOFTMAX_TRACE(stage,index,value) ((void)0)
#endif
namespace ferrule::cuda::core::eager_attention {
constexpr uint64_t workspace_bytes=32ull<<20, provider_bytes=16ull<<20;
inline bool config_literal() { const char *s=std::getenv("CUBLAS_WORKSPACE_CONFIG");return s && std::strcmp(s,":0:0")==0; }
inline const bool startup_config=config_literal();
inline bool plan_valid(const FerruleBf16AttentionPlan &p) {
  if(p.reserved || !p.heads || p.heads>INT_MAX || !p.kv_heads || p.heads%p.kv_heads ||
     !p.dim || p.dim>INT_MAX || !p.max_rows || p.max_rows>p.max_tokens || !p.max_tokens || p.max_tokens>1024)return false;
  uint64_t q,kv,score;
  if(!checked_mul_u64(uint64_t(p.heads)*p.max_rows,uint64_t(p.dim)*2,&q) ||
     !checked_mul_u64(uint64_t(p.heads)*p.max_tokens,uint64_t(p.dim)*2,&kv) ||
     !checked_mul_u64(uint64_t(p.heads)*p.max_rows,uint64_t(p.max_tokens)*2,&score) ||
     q>INT_MAX || kv>INT_MAX || score>INT_MAX)return false;
  uint64_t sizes[]={q,kv,kv,score,score,q,workspace_bytes},end=0;
  for(int i=0;i<7;++i){end=(end+255)&~255ull;if(p.offsets[i]!=end)return false;end+=sizes[i];}
  return p.bytes==end;
}
struct Span { uint64_t begin,end; };
inline bool span(uint64_t begin,uint64_t bytes,uint64_t alignment,Span*out){
  if(!begin||begin%alignment||!bytes||begin>UINT64_MAX-bytes)return false;
  *out={begin,begin+bytes};return true;
}
inline bool overlap(Span a,Span b){return a.begin<b.end&&b.begin<a.end;}
inline bool operands_valid(const FerruleCoreTransformerArgs&a,uint64_t memory,uint64_t bytes){
  for(uint64_t n : {a.block_slots_count,a.block_offsets_count,a.row_sequence_ids_count,a.row_positions_count,a.row_kv_lens_count,a.status_count})if(n>UINT64_MAX/4)return false;
  const uint64_t ptrs[]={a.query_f32,a.append_key_bf16,a.append_value_bf16,a.key_cache_bf16,a.value_cache_bf16,
    a.block_slots_i32,a.block_offsets_i32,a.row_sequence_ids_i32,a.row_positions_i32,a.row_kv_lens_i32,a.output_f32,a.status_i32,memory};
  const uint64_t sizes[]={a.query_bytes,a.append_key_bytes,a.append_value_bytes,a.key_cache_bytes,a.value_cache_bytes,
    a.block_slots_count*4,a.block_offsets_count*4,a.row_sequence_ids_count*4,a.row_positions_count*4,a.row_kv_lens_count*4,a.output_bytes,a.status_count*4,bytes};
  const uint64_t aligns[]={4,2,2,2,2,4,4,4,4,4,4,4,256};Span spans[13];
  const bool app=a.kind==FERRULE_CORE_TRANSFORMER_PAGED_BF16_APPEND_CAUSAL_GQA;
  for(int i=0;i<13;++i){if(!app&&(i==1||i==2)){spans[i]={0,0};continue;}if(!span(ptrs[i],sizes[i],aligns[i],&spans[i]))return false;}
  for(int w : {3,4,10,11,12})for(int r=0;r<13;++r)if(w!=r&&overlap(spans[w],spans[r]))return false;
  return true;
}
inline bool group_valid(const FerruleBf16AttentionPlan&p,const FerruleBf16AttentionGroup&g){return g.rows && g.rows<=p.max_rows && g.tokens && g.tokens<=p.max_tokens && g.rows<=g.tokens;}
inline bool memory_valid(const FerruleBf16AttentionPlan&p,uint64_t memory,uint64_t bytes){return plan_valid(p)&&memory&&memory%256==0&&bytes>=p.bytes&&memory<=UINT64_MAX-bytes;}
inline uint16_t *panel(uint64_t memory,const FerruleBf16AttentionPlan&p,int i){return pointer<uint16_t>(memory+p.offsets[i]);}
inline int matrix(cublasHandle_t h,const FerruleBf16AttentionPlan&p,const FerruleBf16AttentionGroup&g,uint64_t memory,bool pv){
  const float one=1,zero=0;int d=p.dim,m=g.rows,n=g.tokens;
  cublasStatus_t rc;
  if(!pv)rc=cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N,n,m,d,&one,
      panel(memory,p,1),CUDA_R_16BF,d,int64_t(n)*d,panel(memory,p,0),CUDA_R_16BF,d,int64_t(m)*d,
      &zero,panel(memory,p,3),CUDA_R_16BF,n,int64_t(m)*n,p.heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  else rc=cublasGemmStridedBatchedEx(h,CUBLAS_OP_N,CUBLAS_OP_N,d,m,n,&one,
      panel(memory,p,2),CUDA_R_16BF,d,int64_t(n)*d,panel(memory,p,4),CUDA_R_16BF,n,int64_t(m)*n,
      &zero,panel(memory,p,5),CUDA_R_16BF,d,int64_t(m)*d,p.heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return rc==CUBLAS_STATUS_SUCCESS?0:2000+int(rc);
}
// A device gate precedes *all* append writes. Every row must describe the exact
// host-admitted GEMM group; every active page must be addressable. No host reads.
__global__ void validate_group(FerruleCoreTransformerArgs a,FerruleBf16AttentionGroup g) {
  if(a.rows&&*const_pointer<int32_t>(a.status_i32))return;
  uint32_t t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t<g.rows){uint32_t seq,pos,len;uint32_t row=g.row_start+t;
    if(!transformer_row_metadata(a,row,&seq,&pos,&len))return;
    const int32_t first=const_pointer<int32_t>(a.row_positions_i32)[g.row_start];
    if(seq!=g.sequence || len!=g.tokens || pos>=g.tokens || first<0 || uint64_t(first)+t!=pos)
      transformer_error(a,kTransformerMetadataError);
    // Validate exact BF16 query storage before the append barrier, not after
    // cache mutation. Semantic errors latch but are not CUDA completion errors.
    for(uint32_t h=0;h<a.q_heads;++h){
      const float*q=const_pointer<float>(a.query_f32+uint64_t(row)*a.query_row_stride_bytes+uint64_t(h)*a.query_head_stride_bytes);
      for(uint32_t d=0;d<a.head_dim;++d)if(__float_as_uint(q[d])&65535u){transformer_error(a,kTransformerMetadataError);return;}
    }
  }
  if(t<g.tokens){uint64_t off;
    transformer_cache_offset(a,g.sequence,t,a.kv_heads-1,a.head_dim-1,false,&off);
    transformer_cache_offset(a,g.sequence,t,a.kv_heads-1,a.head_dim-1,true,&off);
  }
}
__global__ void append(FerruleCoreTransformerArgs a){
  uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=uint64_t(a.rows)*a.kv_heads*a.head_dim || *const_pointer<int32_t>(a.status_i32))return;
  uint32_t d=i%a.head_dim,h=(i/a.head_dim)%a.kv_heads,row=i/(uint64_t(a.head_dim)*a.kv_heads),seq,pos,len;
  if(!transformer_row_metadata(a,row,&seq,&pos,&len))return;uint64_t ko,vo;
  if(!transformer_cache_offset(a,seq,pos,h,d,false,&ko)||!transformer_cache_offset(a,seq,pos,h,d,true,&vo))return;
  pointer<uint16_t>(a.key_cache_bf16+ko)[0]=const_pointer<uint16_t>(a.append_key_bf16+uint64_t(row)*a.append_key_row_stride_bytes+uint64_t(h)*a.append_key_head_stride_bytes)[d];
  pointer<uint16_t>(a.value_cache_bf16+vo)[0]=const_pointer<uint16_t>(a.append_value_bf16+uint64_t(row)*a.append_value_row_stride_bytes+uint64_t(h)*a.append_value_head_stride_bytes)[d];
}
__global__ void pack(FerruleCoreTransformerArgs a,FerruleBf16AttentionGroup g,uint16_t*q){
  uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=uint64_t(a.q_heads)*g.rows*a.head_dim)return;
  uint32_t d=i%a.head_dim,r=(i/a.head_dim)%g.rows,h=i/(uint64_t(a.head_dim)*g.rows);
  q[i]=0;if(*const_pointer<int32_t>(a.status_i32))return;
  float v=const_pointer<float>(a.query_f32+uint64_t(g.row_start+r)*a.query_row_stride_bytes+uint64_t(h)*a.query_head_stride_bytes)[d];
  // This API consumes exact BF16 expansions, not arbitrary F32 query semantics.
  if((__float_as_uint(v)&65535u)!=0){transformer_error(a,kTransformerMetadataError);return;}
  q[i]=bf16_rne(v);
}
__global__ void gather(FerruleCoreTransformerArgs a,FerruleBf16AttentionGroup g,uint16_t*k,uint16_t*v){
  uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=uint64_t(a.q_heads)*g.tokens*a.head_dim)return;
  k[i]=0;v[i]=0;if(*const_pointer<int32_t>(a.status_i32))return;
  uint32_t d=i%a.head_dim,t=(i/a.head_dim)%g.tokens,h=i/(uint64_t(a.head_dim)*g.tokens),kh=h/(a.q_heads/a.kv_heads);uint64_t ko,vo;
  if(!transformer_cache_offset(a,g.sequence,t,kh,d,false,&ko)||!transformer_cache_offset(a,g.sequence,t,kh,d,true,&vo))return;
  k[i]=const_pointer<uint16_t>(a.key_cache_bf16+ko)[0];v[i]=const_pointer<uint16_t>(a.value_cache_bf16+vo)[0];
}
__global__ void softmax(FerruleCoreTransformerArgs a,FerruleBf16AttentionGroup g,const uint16_t*scores,uint16_t*prob){
  if(!a.rows||*const_pointer<int32_t>(a.status_i32))return;
  uint32_t row=blockIdx.x%g.rows,h=blockIdx.x/g.rows,lane=threadIdx.x;
  uint32_t power=1;while(power<g.tokens)power*=2;uint32_t width=min(power,32u);
  if(lane>=width)return;
  uint64_t base=(uint64_t(h)*g.rows+row)*g.tokens;
  uint32_t pos=const_pointer<int32_t>(a.row_positions_i32)[g.row_start+row];
  float values[32],maximum=-CUDART_INF_F;
  const uint32_t iters=power/width;
  for(uint32_t i=0;i<iters;++i){uint32_t t=lane+i*width;float x=-CUDART_INF_F;
    if(t<g.tokens){x=bf16_round(__fmul_rn(bf16_value(scores[base+t]),a.softmax_scale));
      FERRULE_ATTENTION_SOFTMAX_TRACE(0,base+t,x);
      x=bf16_round(__fadd_rn(x,t<=pos?0.f:bf16_value(0xff7f)));
      FERRULE_ATTENTION_SOFTMAX_TRACE(1,base+t,x);}
    values[i]=x;maximum=fmaxf(maximum,x);
  }
  unsigned mask=__activemask();for(uint32_t s=width/2;s;s/=2)maximum=fmaxf(maximum,__shfl_xor_sync(mask,maximum,s,width));
  float sum=0;for(uint32_t i=0;i<iters;++i){values[i]=expf(__fsub_rn(values[i],maximum));sum=__fadd_rn(sum,values[i]);}
  for(uint32_t s=width/2;s;s/=2)sum=__fadd_rn(sum,__shfl_xor_sync(mask,sum,s,width));
  for(uint32_t i=0;i<iters;++i){uint32_t t=lane+i*width;if(t<g.tokens){float p=values[i]/sum;FERRULE_ATTENTION_SOFTMAX_TRACE(2,base+t,p);prob[base+t]=bf16_rne(p);}}
}
__global__ void scatter(FerruleCoreTransformerArgs a,FerruleBf16AttentionGroup g,const uint16_t*pv){
  uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=uint64_t(a.q_heads)*g.rows*a.head_dim||*const_pointer<int32_t>(a.status_i32))return;
  uint32_t d=i%a.head_dim,r=(i/a.head_dim)%g.rows,h=i/(uint64_t(a.head_dim)*g.rows);
  pointer<float>(a.output_f32+uint64_t(g.row_start+r)*a.output_row_stride_bytes+uint64_t(h)*a.output_head_stride_bytes)[d]=bf16_value(pv[i]);
}
}
extern "C" int ferrule_bf16_attention_create(const FerruleBf16AttentionPlan*p,const FerruleBf16AttentionGroup*groups,uint32_t count,
    uint64_t stream,uint64_t memory,uint64_t bytes,uint64_t*handle,uint64_t*growth,int*version){
  using namespace ferrule::cuda::core::eager_attention;
  if(!handle||!growth||!version)return 1;*handle=0;*growth=0;*version=0;
  if(!p||!groups||!count||!memory_valid(*p,memory,bytes)||!startup_config||!config_literal())return 1;
#if CUBLAS_VER_MAJOR != 12 || CUBLAS_VER_MINOR != 4
  return 2;
#endif
  int device;cudaDeviceProp properties;
  if(cudaGetDevice(&device)!=cudaSuccess||cudaGetDeviceProperties(&properties,device)!=cudaSuccess)return 4;
  if(properties.major!=8||properties.minor!=6)return 2;
  for(uint32_t i=0;i<count;++i)if(!group_valid(*p,groups[i]))return 1;
  auto s=reinterpret_cast<cudaStream_t>(stream);cudaStreamCaptureStatus capture;
  if(cudaStreamIsCapturing(s,&capture)!=cudaSuccess||capture!=cudaStreamCaptureStatusNone)return 1;
  size_t before,total,after;if(cudaMemGetInfo(&before,&total)!=cudaSuccess)return 4;
  if(before<provider_bytes)return 5;
  cublasHandle_t h=nullptr;auto rc=cublasCreate(&h);*handle=reinterpret_cast<uint64_t>(h);
  if(rc!=CUBLAS_STATUS_SUCCESS)return 2000+int(rc);
  if((rc=cublasGetVersion(h,version))!=CUBLAS_STATUS_SUCCESS)return 2000+int(rc);
  if(*version!=120405)return 7;
  if((rc=cublasSetStream(h,s))!=CUBLAS_STATUS_SUCCESS ||
     (rc=cublasSetWorkspace(h,reinterpret_cast<void*>(memory+p->offsets[6]),workspace_bytes))!=CUBLAS_STATUS_SUCCESS ||
     (rc=cublasSetPointerMode(h,CUBLAS_POINTER_MODE_HOST))!=CUBLAS_STATUS_SUCCESS ||
     (rc=cublasSetMathMode(h,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION))!=CUBLAS_STATUS_SUCCESS ||
     (rc=cublasSetAtomicsMode(h,CUBLAS_ATOMICS_ALLOWED))!=CUBLAS_STATUS_SUCCESS)return 2000+int(rc);
  if(cudaMemsetAsync(reinterpret_cast<void*>(memory),0,p->offsets[6],s)!=cudaSuccess)return 4;
  // Materialize the native module and kernel-local resources during admission,
  // not on the first hot launch. Empty work never reads caller operands.
  FerruleCoreTransformerArgs empty{};FerruleBf16AttentionGroup none{};
  validate_group<<<1,256,0,s>>>(empty,none);
  append<<<1,256,0,s>>>(empty);
  pack<<<1,256,0,s>>>(empty,none,panel(memory,*p,0));
  gather<<<1,256,0,s>>>(empty,none,panel(memory,*p,1),panel(memory,*p,2));
  softmax<<<1,32,0,s>>>(empty,none,panel(memory,*p,3),panel(memory,*p,4));
  scatter<<<1,256,0,s>>>(empty,none,panel(memory,*p,5));
  if(cudaGetLastError()!=cudaSuccess)return 4;
  for(int repeat=0;repeat<2;++repeat){
    for(uint32_t i=0;i<count;++i){if(int code=matrix(h,*p,groups[i],memory,false))return code;if(int code=matrix(h,*p,groups[i],memory,true))return code;}
    if(cudaStreamSynchronize(s)!=cudaSuccess)return 9;
    if(cudaMemGetInfo(&after,&total)!=cudaSuccess)return 4;
    // Global memory deltas are diagnostic, never attributable allocation proof.
    *growth=before>after?before-after:0;
  }
  return 0;
}
extern "C" int ferrule_bf16_attention_destroy(uint64_t handle){return int(cublasDestroy(reinterpret_cast<cublasHandle_t>(handle)));}
extern "C" int ferrule_bf16_attention_launch(const FerruleCoreTransformerArgs*a,const FerruleBf16AttentionPlan*p,
    const FerruleBf16AttentionGroup*groups,uint32_t count,uint64_t memory,uint64_t bytes,uint64_t handle){
  using namespace ferrule::cuda::core;using namespace ferrule::cuda::core::eager_attention;
  if(!a||!p||!groups||!count||!handle||!valid_transformer(a)||!memory_valid(*p,memory,bytes)||!operands_valid(*a,memory,bytes)||a->flags!=1||
     (a->kind!=FERRULE_CORE_TRANSFORMER_PAGED_BF16_APPEND_CAUSAL_GQA&&a->kind!=FERRULE_CORE_TRANSFORMER_PAGED_BF16_CAUSAL_GQA)||
     a->q_heads!=p->heads||a->kv_heads!=p->kv_heads||a->head_dim!=p->dim||a->sequences!=count)return 1;
  uint64_t rows=0;for(uint32_t i=0;i<count;++i){if(!group_valid(*p,groups[i])||groups[i].row_start!=rows||groups[i].sequence!=i)return 1;rows+=groups[i].rows;}
  if(rows!=a->rows)return 1;
  auto s=reinterpret_cast<cudaStream_t>(a->stream);auto h=reinterpret_cast<cublasHandle_t>(handle);cudaStream_t hs;cudaStreamCaptureStatus capture;
  if(cublasGetStream(h,&hs)!=CUBLAS_STATUS_SUCCESS||hs!=s||cudaStreamIsCapturing(s,&capture)!=cudaSuccess||capture!=cudaStreamCaptureStatusNone)return 1;
  for(uint32_t i=0;i<count;++i){validate_group<<<(groups[i].tokens+255)/256,256,0,s>>>(*a,groups[i]);if(cudaGetLastError()!=cudaSuccess)return 4;}
  if(a->kind==FERRULE_CORE_TRANSFORMER_PAGED_BF16_APPEND_CAUSAL_GQA){append<<<(uint64_t(a->rows)*a->kv_heads*a->head_dim+255)/256,256,0,s>>>(*a);if(cudaGetLastError()!=cudaSuccess)return 4;}
  else if(a->kind!=FERRULE_CORE_TRANSFORMER_PAGED_BF16_CAUSAL_GQA)return 1;
  for(uint32_t i=0;i<count;++i){const auto g=groups[i];const uint64_t q=uint64_t(p->heads)*g.rows*p->dim,kv=uint64_t(p->heads)*g.tokens*p->dim,n=uint64_t(p->heads)*g.rows*g.tokens;
    pack<<<(q+255)/256,256,0,s>>>(*a,g,panel(memory,*p,0));if(cudaGetLastError()!=cudaSuccess)return 4;
    gather<<<(kv+255)/256,256,0,s>>>(*a,g,panel(memory,*p,1),panel(memory,*p,2));if(cudaGetLastError()!=cudaSuccess)return 4;
    if(int code=matrix(h,*p,g,memory,false))return code;
    FERRULE_ATTENTION_OBSERVE("qk",panel(memory,*p,3),n);
    softmax<<<p->heads*g.rows,32,0,s>>>(*a,g,panel(memory,*p,3),panel(memory,*p,4));if(cudaGetLastError()!=cudaSuccess)return 4;
    FERRULE_ATTENTION_OBSERVE("prob",panel(memory,*p,4),n);
    if(int code=matrix(h,*p,g,memory,true))return code;
    FERRULE_ATTENTION_OBSERVE("pv",panel(memory,*p,5),q);
    scatter<<<(q+255)/256,256,0,s>>>(*a,g,panel(memory,*p,5));if(cudaGetLastError()!=cudaSuccess)return 4;
  }
  return 0;
}

extern "C" int ferrule_bf16_attention_begin(uint64_t stream,uint64_t status){
  if(!status||status%4)return 1;
  return cudaMemsetAsync(reinterpret_cast<void*>(status),0,4,reinterpret_cast<cudaStream_t>(stream))==cudaSuccess?0:4;
}
