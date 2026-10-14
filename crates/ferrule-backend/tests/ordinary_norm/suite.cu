#include "core/tensor_ops.cuh"
#include "core/bf16_boundary.cuh"
#include "norm_observed.cuh"
#include "buffers.cuh"
#include <limits>
using namespace ferrule::cuda::core;
static void legacy(Bf16BoundaryArgs a, Buffer<float> &x, Buffer<float> &w,
                   Buffer<float> &y, cudaStream_t stream) {
  Bf16BoundaryArgs expand{};expand.kind=1;expand.rows=a.rows;expand.width=a.width;
  expand.input=a.input;expand.output=x.addr();expand.stream=a.stream;
  ck(cudaError_t(ferrule_core_bf16_boundary(&expand)));
  expand.rows=1;expand.input=a.other;expand.output=w.addr();
  ck(cudaError_t(ferrule_core_bf16_boundary(&expand)));
  FerruleCoreNormArgs norm{};norm.kind=FERRULE_CORE_NORM_OFFSET_AFFINE_F32;
  norm.rows=a.rows;norm.width=a.width;norm.epsilon=a.epsilon;norm.input=x.addr();norm.weight=w.addr();norm.output=y.addr();
  norm_kernel<<<a.rows,kBlock,0,stream>>>(norm);ck(cudaGetLastError());
  FerruleCoreRowsArgs cast{};cast.kind=FERRULE_CORE_ROWS_F32_TO_BF16_RNE;
  cast.rows=a.rows;cast.heads=1;cast.dimensions=a.width;
  cast.input_f32=y.addr();cast.input_row_stride_bytes=a.width*4;cast.input_head_stride_bytes=a.width*4;
  cast.output_bf16=a.output;cast.output_row_stride_bytes=a.width*2;cast.output_head_stride_bytes=a.width*2;
  rows_f32_to_bf16_rne_kernel<<<(uint64_t(a.rows)*a.width+255)/256,256,0,stream>>>(cast);ck(cudaGetLastError());
}
static void contracts() {
  Bf16BoundaryArgs a{};a.kind=7;a.rows=1;a.width=2048;a.epsilon=1e-6f;
  a.input=0x10000000;a.other=0x20000000;a.output=0x30000000;
  auto reject=[](Bf16BoundaryArgs b){require(ferrule_core_bf16_boundary(&b)!=0,"reject before CUDA");};
  auto b=a;b.rows=0;reject(b);b=a;b.width=0;reject(b);
  b=a;b.width=UINT32_MAX;reject(b);b=a;b.rows=UINT32_MAX;reject(b);
  b=a;b.input=0;reject(b);b=a;b.other=0;reject(b);b=a;b.output=0;reject(b);
  b=a;b.reserved=1;reject(b);b=a;b.input++;reject(b);b=a;b.other++;reject(b);b=a;b.output++;reject(b);
  b=a;b.output=a.input;reject(b);b=a;b.output=a.other;reject(b);
  b=a;b.input=UINT64_MAX-1;reject(b);b=a;b.other=UINT64_MAX-1;reject(b);b=a;b.output=UINT64_MAX-1;reject(b);
  for(float e:{0.0f,-1.0f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}){b=a;b.epsilon=e;reject(b);}
  std::cout<<"native ordinary BF16 norm guards passed without CUDA calls\n";
}
int main(int argc,char**argv){try{
  if(argc==2&&std::string(argv[1])=="--contracts"){contracts();return 0;}
  require(argc==2,"prepared directory");std::string root=argv[1];
  cudaDeviceProp prop;int runtime;ck(cudaGetDeviceProperties(&prop,0));ck(cudaRuntimeGetVersion(&runtime));
  require(prop.major==8&&prop.minor==6&&runtime==12040,"CUDA12.4/sm86");
  cudaFuncAttributes old,newer;ck(cudaFuncGetAttributes(&old,norm_kernel));ck(cudaFuncGetAttributes(&newer,bf16_offset_norm_tree));
  std::cout<<"{\"event\":\"resources\",\"old_norm_registers\":"<<old.numRegs<<",\"new_registers\":"<<newer.numRegs
    <<",\"old_local_bytes\":"<<old.localSizeBytes<<",\"new_local_bytes\":"<<newer.localSizeBytes<<",\"new_shared_bytes\":"<<newer.sharedSizeBytes<<"}\n";
  std::ifstream cases(root+"/cases.tsv");std::string name;uint32_t rows,width;float epsilon;int timed;
  while(cases>>name>>rows>>width>>epsilon>>timed){
    const std::string d=root+"/"+name;uint64_t n=uint64_t(rows)*width;
    Buffer<uint16_t>x(n),w(width),y(n);Buffer<float>xf(n),wf(width),yf(n),trace(3*rows+3*n);
    x.read(d+"/input.bf16");w.read(d+"/weight.bf16");ck(cudaStreamSynchronize(nullptr));
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    Bf16BoundaryArgs a{};a.kind=7;a.rows=rows;a.width=width;a.epsilon=epsilon;a.input=x.addr();a.other=w.addr();a.output=y.addr();a.stream=reinterpret_cast<uint64_t>(stream);
    legacy(a,xf,wf,yf,stream);ck(cudaStreamSynchronize(stream));auto old_y=y.host();y.write(d+"/old.bf16");
    ck(cudaError_t(ferrule_core_bf16_boundary(&a)));ck(cudaStreamSynchronize(stream));auto new_y=y.host();y.write(d+"/native.bf16");
    if(width<=4096){auto block=bf16_rms::block(rows,width);bf16_offset_norm_observed<<<(rows+block.y-1)/block.y,block,0,stream>>>(a,float(1.0/double(width)),trace.p);ck(cudaGetLastError());ck(cudaStreamSynchronize(stream));require(y.host()==new_y,"observer changed output");trace.write(d+"/native-stages.f32");}
    else require(old_y==new_y,"wide fallback stability");
    if(timed){
      cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));
      for(int warm=0;warm<10;++warm){legacy(a,xf,wf,yf,stream);ck(cudaError_t(ferrule_core_bf16_boundary(&a)));}ck(cudaStreamSynchronize(stream));
      for(int pair=0;pair<30;++pair)for(int order=0;order<2;++order){bool before=(pair+order)%2==0;
        ck(cudaEventRecord(start,stream));if(before)legacy(a,xf,wf,yf,stream);else ck(cudaError_t(ferrule_core_bf16_boundary(&a)));ck(cudaEventRecord(stop,stream));ck(cudaEventSynchronize(stop));
        float ms;ck(cudaEventElapsedTime(&ms,start,stop));require(y.host()==(before?old_y:new_y),"timed result drift");x.immutable();w.immutable();
        std::cout<<"{\"event\":\"timing\",\"pair\":"<<pair<<",\"order\":"<<order<<",\"variant\":\""<<(before?"old4":"new1")<<"\",\"us\":"<<ms*1000<<"}\n";
      }ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));
    }
    x.immutable();w.immutable();y.host();trace.host();xf.host();wf.host();yf.host();ck(cudaStreamDestroy(stream));
    std::cout<<"{\"event\":\"case\",\"name\":\""<<name<<"\",\"guards\":true,\"inputs_immutable\":true}\n";
  }
  require(cases.eof(),"case table");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
