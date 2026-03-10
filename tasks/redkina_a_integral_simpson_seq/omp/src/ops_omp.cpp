#include "redkina_a_integral_simpson_seq/omp/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "redkina_a_integral_simpson_seq/common/include/common.hpp"

namespace redkina_a_integral_simpson_seq {

namespace {

inline double GetCoeff(int idx, int n) {
  if (idx == 0 || idx == n) {
    return 1.0;
  }
  return (idx % 2 == 1) ? 4.0 : 2.0;
}

}  // namespace

RedkinaAIntegralSimpsonOMP::RedkinaAIntegralSimpsonOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool RedkinaAIntegralSimpsonOMP::ValidationImpl() {
  const auto &in = GetInput();
  size_t dim = in.a.size();

  if (dim == 0 || in.n.size() != dim || in.b.size() != dim) {
    return false;
  }

  for (size_t i = 0; i < dim; ++i) {
    if (in.a[i] >= in.b[i]) {
      return false;
    }
    if (in.n[i] <= 0 || in.n[i] % 2 != 0) {
      return false;
    }
  }

  return static_cast<bool>(in.func);
}

bool RedkinaAIntegralSimpsonOMP::PreProcessingImpl() {
  const auto &in = GetInput();

  func_ = in.func;
  a_ = in.a;
  b_ = in.b;
  n_ = in.n;

  result_ = 0.0;

  return true;
}

bool RedkinaAIntegralSimpsonOMP::RunImpl() {
  const size_t dim = a_.size();

  std::vector<double> h(dim);
  for (size_t i = 0; i < dim; ++i) {
    h[i] = (b_[i] - a_[i]) / static_cast<double>(n_[i]);
  }

  double h_prod = 1.0;
  for (size_t i = 0; i < dim; ++i) {
    h_prod *= h[i];
  }

  long long total_points = 1;
  for (size_t i = 0; i < dim; ++i) {
    total_points *= (static_cast<long long>(n_[i]) + 1);
  }

  double sum = 0.0;

#pragma omp parallel for default(none) reduction(+ : sum) shared(total_points)
  for (long long linear_idx = 0; linear_idx < total_points; ++linear_idx) {
    std::vector<int> indices(dim);
    std::vector<double> point(dim);

    long long tmp = linear_idx;
    double weight_prod = 1.0;

    for (int d = static_cast<int>(dim) - 1; d >= 0; --d) {
      int size_d = n_[d] + 1;
      indices[d] = static_cast<int>(tmp % size_d);
      tmp /= size_d;
    }

    for (size_t d = 0; d < dim; ++d) {
      point[d] = a_[d] + indices[d] * h[d];
      weight_prod *= GetCoeff(indices[d], n_[d]);
    }

    sum += weight_prod * func_(point);
  }

  double denominator = 1.0;
  for (size_t i = 0; i < dim; ++i) {
    denominator *= 3.0;
  }

  result_ = (h_prod / denominator) * sum;

  return true;
}

bool RedkinaAIntegralSimpsonOMP::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace redkina_a_integral_simpson_seq
