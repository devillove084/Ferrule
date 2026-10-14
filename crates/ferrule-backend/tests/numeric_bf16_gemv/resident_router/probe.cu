#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
void ck(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
void cb(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cublas status "+std::to_string(s));}
std::vector<uint16_t> read(std::string p,size_t n){std::vector<uint16_t>x(n);std::ifstream f(p,std::ios::binary);if(!f.read((char*)x.data(),n*2)||f.peek()!=EOF)throw std::runtime_error("extent "+p);return x;}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("fixture");std::string dir=argv[1];cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));int rt;ck(cudaRuntimeGetVersion(&rt));if(rt!=12040||prop.major!=8||prop.minor!=6)throw std::runtime_error("CUDA12.4 sm86 required");
 cudaStream_t stream;ck(cudaStreamCreate(&stream));cublasHandle_t h;cb(cublasCreate(&h));int version;cb(cublasGetVersion(h,&version));if(version!=120405)throw std::runtime_error("cuBLAS120405");void*ws;ck(cudaMalloc(&ws,32<<20));cb(cublasSetStream(h,stream));cb(cublasSetWorkspace(h,ws,32<<20));cb(cublasSetMathMode(h,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));cb(cublasSetAtomicsMode(h,CUBLAS_ATOMICS_NOT_ALLOWED));
 std::ifstream cases(dir+"/cases.txt");std::string name;int m,n,k;while(cases>>name>>m>>n>>k){auto x=read(dir+"/"+name+".input.bf16",m*k),w=read(dir+"/"+name+".weight.bf16",n*k),expected=read(dir+"/"+name+".hf.bf16",m*n);uint16_t *xd,*wd,*yd;ck(cudaMalloc(&xd,x.size()*2));ck(cudaMalloc(&wd,w.size()*2));ck(cudaMalloc(&yd,m*n*2));ck(cudaMemcpy(xd,x.data(),x.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(wd,w.data(),w.size()*2,cudaMemcpyHostToDevice));float alpha=1,beta=0;cb(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,wd,CUDA_R_16BF,k,xd,CUDA_R_16BF,k,&beta,yd,CUDA_R_16BF,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));ck(cudaStreamSynchronize(stream));std::vector<uint16_t> y(m*n);ck(cudaMemcpy(y.data(),yd,y.size()*2,cudaMemcpyDeviceToHost));size_t bad=0;for(size_t i=0;i<y.size();++i)bad+=y[i]!=expected[i];std::ofstream out(dir+"/"+name+".direct.bf16",std::ios::binary);out.write((char*)y.data(),y.size()*2);std::cout<<"{\"case\":\""<<name<<"\",\"words\":"<<y.size()<<",\"unequal\":"<<bad<<",\"n32_diagnostic_only\":"<<(n==32?"true":"false")<<"}\n";ck(cudaFree(xd));ck(cudaFree(wd));ck(cudaFree(yd));}
 cb(cublasDestroy(h));ck(cudaFree(ws));ck(cudaStreamDestroy(stream));
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
