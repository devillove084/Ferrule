#pragma once
#include <cuda_runtime.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
static void ck(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
template <class T> struct Buffer {
  T *base, *p; size_t n; std::vector<T> original;
  explicit Buffer(size_t count) : n(count) {
    ck(cudaMalloc(&base, (n+32)*sizeof(T))); p=base+16;
    ck(cudaMemset(base,0x5a,(n+32)*sizeof(T)));
  }
  ~Buffer() { cudaFree(base); }
  uint64_t addr() const { return reinterpret_cast<uint64_t>(p); }
  void read(const std::string &path) {
    original.resize(n); std::ifstream f(path,std::ios::binary);
    require(bool(f.read(reinterpret_cast<char *>(original.data()),n*sizeof(T))) && f.peek()==EOF,"input extent");
    ck(cudaMemcpy(p,original.data(),n*sizeof(T),cudaMemcpyHostToDevice));
  }
  std::vector<T> host() {
    std::vector<T> data(n+32);ck(cudaMemcpy(data.data(),base,data.size()*sizeof(T),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<16*sizeof(T);++i)
      require(reinterpret_cast<unsigned char *>(data.data())[i]==0x5a &&
              reinterpret_cast<unsigned char *>(data.data()+n+16)[i]==0x5a,"guard");
    return {data.begin()+16,data.end()-16};
  }
  void immutable() { auto v=host();require(!memcmp(v.data(),original.data(),n*sizeof(T)),"input modified"); }
  void write(const std::string &path) {
    auto v=host();std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char *>(v.data()),n*sizeof(T));require(bool(f),"write");
  }
};
