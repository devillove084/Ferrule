// Test-only bounded CUDA12.4 native GEMV candidate, never a dispatch override.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#define CK(x) do { auto s=(x); if(s!=0){fprintf(stderr,"%s status=%d\n",#x,int(s));std::exit(2);} }while(0)
template<class T> std::vector<T> read(const std::string& p,size_t count){
 std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f || size_t(f.tellg())!=count*sizeof(T))std::exit(3);
 f.seekg(0);std::vector<T> v(count);f.read(reinterpret_cast<char*>(v.data()),count*sizeof(T));if(!f)std::exit(4);return v;
}
struct Buffer {void* p=nullptr;explicit Buffer(size_t bytes){CK(cudaMalloc(&p,bytes));}~Buffer(){if(p)cudaFree(p);}};
template<class T> void upload(Buffer& b,const std::vector<T>& v){CK(cudaMemcpy(b.p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
__global__ void decode(const unsigned char* w,const __nv_bfloat16* s,__nv_bfloat16* out,int first,int count,int k){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count*k;i+=gridDim.x*blockDim.x){
  int row=first+i/k,col=i%k;__nv_fp8_e4m3 raw;raw.__x=w[row*k+col];
  out[i]=__float2bfloat16_rn(__fmul_rn(float(raw),__bfloat162float(s[(row/128)*(k/128)+col/128])));
 }
}
static uint16_t rne(float x){uint32_t u;std::memcpy(&u,&x,4);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
static float value(uint16_t x){uint32_t u=uint32_t(x)<<16;float f;std::memcpy(&f,&u,4);return f;}
int main(int argc,char**argv){
 if(argc!=5)return 3;std::string prefix=std::string(argv[1])+"/"+argv[2];
 const int n=std::stoi(argv[3]),k=std::stoi(argv[4]);if(!((n==2048 && (k==4096 || k==512)) || (n==512 && k==2048)))return 3;
 auto raw=read<unsigned char>(prefix+".weight.fp8",n*k);auto scales=read<uint16_t>(prefix+".scale.bf16",n/128*k/128);
 auto input=read<uint16_t>(prefix+".input.bf16",k),expected=read<uint16_t>(prefix+".hf.bf16",n);
 Buffer w(raw.size()),s(scales.size()*2),x(k*2),y(n*4),yb(n*2);upload(w,raw);upload(s,scales);upload(x,input);
 cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));size_t before,total,after;CK(cudaMemGetInfo(&before,&total));cublasHandle_t handle;CK(cublasCreate(&handle));CK(cublasSetStream(handle,stream));CK(cublasSetWorkspace(handle,nullptr,0));CK(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));
 CK(cudaMemGetInfo(&after,&total));printf("{\"event\":\"handle\",\"device_growth_bytes\":%zu}\n",before>after?before-after:0);
 float alpha=1,beta=0;int version;CK(cublasGetVersion(handle,&version));
 for(int tile: {257,1023,2048}){
  Buffer stage(size_t(tile)*k*2);
  for(int mode: {0,1}){
   CK(cudaMemGetInfo(&before,&total));
   for(int first=0;first<n;first+=tile){int count=std::min(tile,n-first);decode<<<std::min(65535,(count*k+255)/256),256,0,stream>>>(static_cast<unsigned char*>(w.p),static_cast<__nv_bfloat16*>(s.p),static_cast<__nv_bfloat16*>(stage.p),first,count,k);CK(cudaGetLastError());
    if(mode==0){CK(cublasTSSgemvStridedBatched(handle,CUBLAS_OP_T,k,count,&alpha,static_cast<__nv_bfloat16*>(stage.p),k,0,static_cast<__nv_bfloat16*>(x.p),1,0,&beta,static_cast<float*>(y.p)+first,1,0,1));}
    else {CK(cublasTSTgemvStridedBatched(handle,CUBLAS_OP_T,k,count,&alpha,static_cast<__nv_bfloat16*>(stage.p),k,0,static_cast<__nv_bfloat16*>(x.p),1,0,&beta,static_cast<__nv_bfloat16*>(yb.p)+first,1,0,1));}
   }
   CK(cudaStreamSynchronize(stream));CK(cudaMemGetInfo(&after,&total));printf("{\"event\":\"submit_memory\",\"device_growth_bytes\":%zu}\n",before>after?before-after:0);std::vector<float> a(n);std::vector<uint16_t> b(n);
   if(mode==0){CK(cudaMemcpy(a.data(),y.p,n*4,cudaMemcpyDeviceToHost));for(int i=0;i<n;++i)b[i]=rne(a[i]);}
   else {CK(cudaMemcpy(b.data(),yb.p,n*2,cudaMemcpyDeviceToHost));}
   int unequal=0;double max_abs=0;
   for(int i=0;i<n;++i)if(b[i]!=expected[i]){++unequal;max_abs=std::max(max_abs,std::abs(double(value(b[i]))-value(expected[i])));printf("{\"event\":\"difference\",\"tile\":%d,\"output_type\":\"%s\",\"index\":%d,\"actual\":%.17g,\"hf\":%.17g}\n",tile,mode?"BF16":"F32",i,double(value(b[i])),double(value(expected[i])));}
   printf("{\"event\":\"comparison\",\"cublas_version\":%d,\"tile\":%d,\"output_type\":\"%s\",\"unequal\":%d,\"elements\":%d,\"max_abs\":%.17g,\"staging_bytes\":%zu}\n",version,tile,mode?"BF16":"F32",unequal,n,max_abs,size_t(tile)*k*2);fflush(stdout);
  }
 }
 CK(cublasDestroy(handle));CK(cudaStreamDestroy(stream));return 0;
}
