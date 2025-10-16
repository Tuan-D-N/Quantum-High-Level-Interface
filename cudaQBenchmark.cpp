// bench_simple_cudaq.cuqq
#include <cudaq.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

static inline double ms(auto d){ using namespace std::chrono; return duration_cast<duration<double,std::milli>>(d).count(); }

__qpu__ void x_cx_pass(cudaq::qview<> qs){
  // X down the register
  for (std::size_t q=0; q<qs.size(); ++q) x(qs[q]);
  // CX down the register
  for (std::size_t q=0; q+1<qs.size(); ++q) cx(qs[q], qs[q+1]);
}

__qpu__ void kernel_n_passes(std::size_t n, int passes){
  auto qs = cudaq::qvector(n); // |0...0>
  for (int r=0;r<passes;++r) x_cx_pass(qs);
}

int main(int argc, char** argv){
  int n_min=16, n_max=28, step=2, repeats=3, passes=10;
  if (argc>=6){ n_min=std::atoi(argv[1]); n_max=std::atoi(argv[2]); step=std::atoi(argv[3]); repeats=std::atoi(argv[4]); passes=std::atoi(argv[5]); }

  // If you have the NVIDIA GPU target: cudaq::set_target("nvidia");

  std::cout << "backend,n,passes,repeats,mean_ms,stdev_ms\n";
  for (int n=n_min; n<=n_max; n+=step){
    (void)cudaq::sample(kernel_n_passes, (std::size_t)n, passes); // warm-up

    std::vector<double> v;
    for (int r=0; r<repeats; ++r){
      auto t0 = std::chrono::high_resolution_clock::now();
      (void)cudaq::sample(kernel_n_passes, (std::size_t)n, passes);
      v.push_back(ms(std::chrono::high_resolution_clock::now() - t0));
    }
    double mu = std::accumulate(v.begin(),v.end(),0.0)/v.size();
    double s2=0; for(double x:v){ double d=x-mu; s2+=d*d; }
    double sd = std::sqrt(s2/std::max(1,(int)v.size()-1));
    std::cout << "cudaq,"<<n<<","<<passes<<","<<repeats<<","<<std::fixed<<std::setprecision(3)<<mu<<","<<sd<<"\n";
  }
  return 0;
}
