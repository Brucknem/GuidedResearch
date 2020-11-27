#include <iostream>
#include <math.h>

// // Kernel function to add the elements of two arrays
// __global__
// void add(int n, float *x, float *y)
// {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;
//   for (int i = index; i < n; i += stride)
//     y[i] = x[i] + y[i];
// }

// int main(void)
// {
//     int N = 1 << 20;
//     float *x, *y;

//     // Allocate Unified Memory – accessible from CPU or GPU
//     cudaMallocManaged(&x, N * sizeof(float));
//     cudaMallocManaged(&y, N * sizeof(float));

//     // initialize x and y arrays on the host
//     for (int i = 0; i < N; i++)
//     {
//         x[i] = 1.0f;
//         y[i] = 2.0f;
//     }

//     int blockSize = 256;
//     int numBlocks = (N + blockSize - 1) / blockSize;
//     add<<<numBlocks, blockSize>>>(N, x, y);

//     // Wait for GPU to finish before accessing on host
//     cudaDeviceSynchronize();

//     // Check for errors (all values should be 3.0f)
//     float maxError = 0.0f;
//     for (int i = 0; i < N; i++){
//         maxError = fmax(maxError, fabs(y[i] - 3.0f));
//     }

//     std::cout << "Max error: " << maxError << std::endl;

//     // Free memory
//     cudaFree(x);
//     cudaFree(y);

//     return 0;
// }

#include "ceres/ceres.h"
#include "glog/logging.h"
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.
struct CostFunctor {
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double x = 0.5;
  const double initial_x = x;
  // Build the problem.
  Problem problem;
  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  CostFunction* cost_function =
      new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
  problem.AddResidualBlock(cost_function, nullptr, &x);
  // Run the solver!
  Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x << " -> " << x << "\n";
  return 0;
}