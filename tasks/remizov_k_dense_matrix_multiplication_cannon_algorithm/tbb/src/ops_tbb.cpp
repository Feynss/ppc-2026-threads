#include "remizov_k_dense_matrix_multiplication_cannon_algorithm/tbb/include/ops_tbb.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#include "remizov_k_dense_matrix_multiplication_cannon_algorithm/common/include/common.hpp"

namespace remizov_k_dense_matrix_multiplication_cannon_algorithm {

RemizovKDenseMatrixMultiplicationCannonAlgorithmTbb::RemizovKDenseMatrixMultiplicationCannonAlgorithmTbb(
    const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

}  // namespace remizov_k_dense_matrix_multiplication_cannon_algorithm
