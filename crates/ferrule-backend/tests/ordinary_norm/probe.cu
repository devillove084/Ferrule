#include "core/tensor_ops.cuh"
#include "core/recurrent_bf16.cuh"
#include "old_observed.cuh"
#include "buffers.cuh"
using namespace ferrule::cuda::core;
__global__ void candidate(FerruleCoreBf16GatedNormArgs a, float *trace) {
  __shared__ float sums[512];
  const uint32_t row=blockIdx.x*blockDim.y+threadIdx.y;
  const uint32_t lane=threadIdx.x, offset=threadIdx.y*blockDim.x;
  sums[offset+lane]=row<a.rows ? hf_bf16::gated_norm_lane_sum(a,row,lane,blockDim.x) : 0;
  __syncthreads();
  for(uint32_t s=blockDim.x/2;s;s/=2) {
    if(lane<s)sums[offset+lane]=__fadd_rn(sums[offset+lane],sums[offset+lane+s]);
    __syncthreads();
  }
  if(row>=a.rows)return;
  float mean=__fmul_rn(sums[offset],float(1.0/double(a.width)));
  float added=__fadd_rn(mean,a.epsilon), inv=rsqrtf(added);
  if(lane==0){trace[row]=mean;trace[a.rows+row]=added;trace[2*a.rows+row]=inv;}
  const uint64_t n=uint64_t(a.rows)*a.width,base=uint64_t(row)*a.width;
  for(uint32_t d=lane;d<a.width;d+=blockDim.x){
    float normalized=__fmul_rn(hf_bf16::load(a.input_bf16,base+d),inv);
    float affine=__fadd_rn(1.0f,hf_bf16::load(a.weight_bf16,d));
    float value=__fmul_rn(normalized,affine);
    trace[3*a.rows+base+d]=normalized;trace[3*a.rows+n+base+d]=affine;trace[3*a.rows+2*n+base+d]=value;
    pointer<uint16_t>(a.output_bf16)[base+d]=bf16_rne(value);
  }
}
int main(int argc,char **argv){try{
  require(argc==2,"fixture directory");std::string d=argv[1];
  cudaDeviceProp prop;int runtime;ck(cudaGetDeviceProperties(&prop,0));ck(cudaRuntimeGetVersion(&runtime));
  require(prop.major==8&&prop.minor==6&&runtime==12040,"CUDA12.4 sm86");
  Buffer<uint16_t>x(2048),w(2048),out(2048);Buffer<float>xf(2048),wf(2048),old(2048),observed(2048),trace(3+3*2048);
  x.read(d+"/input.bf16");w.read(d+"/weight.bf16");xf.read(d+"/input.f32");wf.read(d+"/weight.f32");
  FerruleCoreNormArgs a{};a.kind=FERRULE_CORE_NORM_OFFSET_AFFINE_F32;a.rows=1;a.width=2048;a.epsilon=1e-6f;a.input=xf.addr();a.weight=wf.addr();a.output=old.addr();
  norm_kernel<<<1,kBlock>>>(a);ck(cudaGetLastError());old.write(d+"/old.f32");
  a.output=observed.addr();old_norm_observed<<<1,kBlock>>>(a,trace.p);ck(cudaGetLastError());
  auto oldv=old.host(), obv=observed.host();require(!memcmp(oldv.data(),obv.data(),oldv.size()*4),"old observer changed output");trace.write(d+"/old-stages.f32");
  FerruleCoreBf16GatedNormArgs b{};b.rows=1;b.width=2048;b.epsilon=1e-6f;b.input_bf16=x.addr();b.weight_bf16=w.addr();b.output_bf16=out.addr();
  candidate<<<1,hf_bf16::gated_norm_block(1,2048)>>>(b,trace.p);ck(cudaGetLastError());out.write(d+"/candidate.bf16");trace.write(d+"/candidate-stages.f32");
  x.immutable();w.immutable();xf.immutable();wf.immutable();std::cout<<"old observer exact; test-only candidate and old stages saved; guards pass\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
