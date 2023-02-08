/***************************************************************************
 *
 *  @license
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename tbsv.hpp
 *
 **************************************************************************/

#ifndef TBSV_HPP
#define TBSV_HPP
#include "operations/blas2_trees.h"
#include "operations/blas_operators.hpp"
#include "views/view_sycl.hpp"
#include <stdexcept>
#include <vector>
namespace blas {

/**
 * @struct Tbsv
 * @brief Tree node representing a triangular band matrix_ vector_
 * multiplication.
 */
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE
Tbsv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
     is_unitdiag>::Tbsv(lhs_t &_l, matrix_t &_matrix,
                        typename Tbsv<lhs_t, matrix_t, vector_t, local_range,
                                      is_upper, is_transposed,
                                      is_unitdiag>::index_t &_k,
                        vector_t &_vector)
    : lhs_(_l), matrix_(_matrix), vector_(_vector), k_(_k) {}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE typename Tbsv<lhs_t, matrix_t, vector_t, local_range, is_upper,
                               is_transposed, is_unitdiag>::index_t
Tbsv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
     is_unitdiag>::get_size() const {
  return matrix_.get_size();
}
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE bool
Tbsv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
     is_unitdiag>::valid_thread(cl::sycl::nd_item<1> ndItem) const {
  // Valid threads are established by ::eval.
  return true;
}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
template <typename local_memory_t>
SYCL_BLAS_INLINE typename Tbsv<lhs_t, matrix_t, vector_t, local_range, is_upper,
                               is_transposed, is_unitdiag>::value_t
Tbsv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
     is_unitdiag>::eval(local_memory_t local_mem, cl::sycl::nd_item<1> ndItem) {
  const index_t global_idx = ndItem.get_global_id(0);

  // copy lhs_ local memory + sync thread
  auto lhs_l = local_mem.localAcc;
  lhs_l[global_idx] = lhs_.eval(global_idx);

  // ----

  // j -> lhs_idx
  for (index_t lhs_idx = 0; lhs_idx < lhs_.get_size(); ++lhs_idx) {
    const index_t s_idx = ndItem.get_local_id(0);
    const index_t k_end = cl::sycl::min(k_, lhs_idx + k_ + 1);

    if (!is_unitdiag && !s_idx) lhs_l[lhs_idx] /= matrix_.eval(0, lhs_idx);
    // sync the WI
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    if ((s_idx > lhs_idx) && (s_idx < k_end)) {
      const index_t K = s_idx - lhs_idx;
      const index_t J = lhs_idx;

      const value_t A =
          (is_unitdiag && (K == 0)) ? value_t(1) : matrix_.eval(K, J);

      // x solution
      lhs_l[s_idx] = lhs_l[s_idx] - ProductOperator::eval(A, lhs_l[lhs_idx]);
    }
    // sync the WI
    ndItem.barrier(cl::sycl::access::fence_space::local_space);
  }

  // ----

  // copy back to local memory lhs_ --> WI sync in above loop
  lhs_.eval(global_idx) = lhs_l[global_idx];

  return 0;
}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE void
Tbsv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
     is_unitdiag>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  matrix_.bind(h);
  vector_.bind(h);
}
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE void
Tbsv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
     is_unitdiag>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  matrix_.adjust_access_displacement();
  vector_.adjust_access_displacement();
}

}  // namespace blas
#endif
