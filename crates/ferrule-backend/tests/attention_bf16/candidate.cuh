// Test-only candidate: block-local F32 QK tree and pinned persistent softmax.
// No reference values are read by this kernel.
__global__ void eager_candidate(FerruleCoreTransformerArgs a, float *trace) {
  __shared__ float sums[128], scores[1024], probabilities[1024];
  __shared__ int failed;
  const uint32_t h=blockIdx.x%a.q_heads,row=blockIdx.x/a.q_heads,lane=threadIdx.x;
  const uint32_t kh=h/(a.q_heads/a.kv_heads);
  uint32_t seq,pos,len;
  if(!transformer_row_metadata(a,row,&seq,&pos,&len))return;
  if(a.kind==FERRULE_CORE_TRANSFORMER_PAGED_BF16_APPEND_CAUSAL_GQA)len=max(len,pos+1);
  uint32_t visible=min(len,pos+1);
  if(lane==0)failed=0;
  __syncthreads();
  const uint64_t q=uint64_t(row)*a.query_row_stride_bytes+uint64_t(h)*a.query_head_stride_bytes;
  for(uint32_t t=0;t<visible;++t){
    float dot=0;
    for(uint32_t d=lane;d<a.head_dim;d+=128){uint64_t off;
      if(!transformer_cache_offset(a,seq,t,kh,d,false,&off)){atomicExch(&failed,1);break;}
      dot=__fmaf_rn(const_pointer<float>(a.query_f32+q)[d],bf16_value(const_pointer<uint16_t>(a.key_cache_bf16+off)[0]),dot);
    }
    sums[lane]=dot;__syncthreads();
    for(uint32_t s=64;s;s/=2){if(lane<s)sums[lane]=__fadd_rn(sums[lane],sums[lane+s]);__syncthreads();}
    if(lane==0){float raw=bf16_round(sums[0]);scores[t]=bf16_round(__fmul_rn(raw,a.softmax_scale));if(trace){trace[(uint64_t(row)*a.q_heads+h)*len+t]=raw;}}
    __syncthreads();
  }
  if(failed)return;
  uint32_t pow2=1;while(pow2<len)pow2*=2;uint32_t warp=pow2<32?pow2:32;
  if(lane<warp){
    float maximum=-CUDART_INF_F;
    for(uint32_t t=lane;t<visible;t+=warp)maximum=fmaxf(maximum,scores[t]);
    unsigned mask=__activemask();
    for(uint32_t s=warp/2;s;s/=2)maximum=fmaxf(maximum,__shfl_xor_sync(mask,maximum,s,warp));
    float sum=0;
    for(uint32_t t=lane;t<pow2;t+=warp){float e=t<visible?expf(__fsub_rn(scores[t],maximum)):0; if(t<visible)probabilities[t]=e;sum=__fadd_rn(sum,e);}
    for(uint32_t s=warp/2;s;s/=2)sum=__fadd_rn(sum,__shfl_xor_sync(mask,sum,s,warp));
    for(uint32_t t=lane;t<visible;t+=warp){float p=probabilities[t]/sum;if(trace){const uint64_t n=uint64_t(a.rows)*a.q_heads*len;trace[n+(uint64_t(row)*a.q_heads+h)*len+t]=p;}
      probabilities[t]=bf16_round(p);
    }
  }
  __syncthreads();
  for(uint32_t d=lane;d<a.head_dim;d+=128){float out=0;
    for(uint32_t t=0;t<visible;++t){uint64_t off;if(!transformer_cache_offset(a,seq,t,kh,d,true,&off))return;out=__fmaf_rn(probabilities[t],bf16_value(const_pointer<uint16_t>(a.value_cache_bf16+off)[0]),out);}
    const uint64_t offset=uint64_t(row)*a.output_row_stride_bytes+uint64_t(h)*a.output_head_stride_bytes+uint64_t(d)*4;
    pointer<float>(a.output_f32+offset)[0]=out;
  }
}
