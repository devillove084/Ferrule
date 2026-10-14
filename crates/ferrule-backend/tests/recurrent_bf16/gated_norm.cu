// Dedicated BF16 gated-norm boundary and matched event regression.
#include "core/recurrent_bf16.cuh"
#include "prior.cuh"
#include "norm_observed.cuh"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace ferrule::cuda::core;
static void ck(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Buffer {
  T *base,*p;size_t n;std::vector<T> original;
  Buffer(size_t count):n(count){ck(cudaMalloc(&base,(n+32)*sizeof(T)));p=base+16;ck(cudaMemset(base,0x5a,(n+32)*sizeof(T)));}
  ~Buffer(){cudaFree(base);}
  uint64_t addr()const{return reinterpret_cast<uint64_t>(p);}
  void read(std::string name){original.resize(n);std::ifstream f(name,std::ios::binary);if(!f.read((char*)original.data(),n*sizeof(T))||f.peek()!=EOF)throw std::runtime_error("input extent "+name);ck(cudaMemcpy(p,original.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
  std::vector<T> host(){std::vector<T> data(n+32);ck(cudaMemcpy(data.data(),base,data.size()*sizeof(T),cudaMemcpyDeviceToHost));for(size_t i=0;i<16*sizeof(T);++i)if(((unsigned char*)data.data())[i]!=0x5a||((unsigned char*)(data.data()+n+16))[i]!=0x5a)throw std::runtime_error("guard");return {data.begin()+16,data.end()-16};}
  void immutable(){auto v=host();if(memcmp(v.data(),original.data(),n*sizeof(T)))throw std::runtime_error("input modified");}
  void write(std::string name){auto v=host();std::ofstream f(name,std::ios::binary);f.write((char*)v.data(),n*sizeof(T));if(!f)throw std::runtime_error("write");}
};
static void launch(FerruleCoreBf16GatedNormArgs &a,bool old){ck(cudaError_t(old?prior_core_bf16_gated_norm_launch(&a):ferrule_core_bf16_gated_norm_launch(&a)));}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("prepared directory");std::string dir=argv[1];
 cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));int runtime;ck(cudaRuntimeGetVersion(&runtime));if(runtime!=12040||prop.major!=8||prop.minor!=6)throw std::runtime_error("CUDA12.4/sm86 required");
 cudaFuncAttributes old_attr,new_attr;ck(cudaFuncGetAttributes(&old_attr,prior_hf_bf16::gated_norm));ck(cudaFuncGetAttributes(&new_attr,hf_bf16::gated_norm_tree));
 std::cout<<"{\"event\":\"resources\",\"old_registers\":"<<old_attr.numRegs<<",\"new_registers\":"<<new_attr.numRegs<<",\"old_local_bytes\":"<<old_attr.localSizeBytes<<",\"new_local_bytes\":"<<new_attr.localSizeBytes<<",\"shared_bytes\":"<<new_attr.sharedSizeBytes<<"}\n";
 std::ifstream cases(dir+"/cases.tsv");std::string name;uint32_t rows,width;float epsilon;int timed;
 while(cases>>name>>rows>>width>>epsilon>>timed){
  std::string d=dir+"/"+name;size_t n=uint64_t(rows)*width;
  Buffer<uint16_t>x(n),z(n),w(width),y(n);Buffer<float>trace(2*rows+4*n);
  x.read(d+"/input.bf16");z.read(d+"/gate.bf16");w.read(d+"/weight.bf16");
  ck(cudaStreamSynchronize(nullptr));cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  FerruleCoreBf16GatedNormArgs a{rows,width,epsilon,0,x.addr(),w.addr(),z.addr(),y.addr(),reinterpret_cast<uint64_t>(stream)};
  launch(a,true);ck(cudaStreamSynchronize(stream));auto old=y.host();y.write(d+"/old.bf16");
  launch(a,false);ck(cudaStreamSynchronize(stream));auto current=y.host();y.write(d+"/native.bf16");
  if(width<=4096){auto block=hf_bf16::gated_norm_block(rows,width);hf_bf16::gated_norm_observed<<<(rows+block.y-1)/block.y,block,0,stream>>>(a,float(1.0/double(width)),trace.p);ck(cudaGetLastError());ck(cudaStreamSynchronize(stream));if(y.host()!=current)throw std::runtime_error("observer changed output "+name);trace.write(d+"/native-stages.f32");}
  else if(old!=current)throw std::runtime_error("fallback changed");
  if(timed){
   cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));
   for(int i=0;i<20;++i){launch(a,true);launch(a,false);}ck(cudaStreamSynchronize(stream));
   for(int sample=0;sample<30;++sample)for(int order=0;order<2;++order){bool before=((sample+order)%2)==0;float total=0;
    for(int i=0;i<8;++i){ck(cudaEventRecord(start,stream));launch(a,before);ck(cudaEventRecord(stop,stream));ck(cudaEventSynchronize(stop));float ms;ck(cudaEventElapsedTime(&ms,start,stop));total+=ms*1000;}
    if(y.host()!=(before?old:current))throw std::runtime_error("timed output drift");x.immutable();z.immutable();w.immutable();
    std::cout<<"{\"event\":\"sample\",\"case\":\""<<name<<"\",\"sample\":"<<sample<<",\"order\":"<<order<<",\"variant\":\""<<(before?"old":"new")<<"\",\"us\":"<<total/8<<"}\n";
   }ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));
  }
  x.immutable();z.immutable();w.immutable();trace.host();ck(cudaStreamDestroy(stream));
  std::cout<<"{\"event\":\"case\",\"name\":\""<<name<<"\",\"rows\":"<<rows<<",\"width\":"<<width<<",\"guards\":true,\"input_immutable\":true}\n";
 }
 if(!cases.eof())throw std::runtime_error("cases parse");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
