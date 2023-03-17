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
 *  @filename trsv.hpp
 *
 **************************************************************************/

#ifndef TRSV_2_HPP
#define TRSV_2_HPP
#include "operations/blas2_trees.h"
namespace blas {

/**
 * @struct Trsv_2
 * @brief Tree node representing a triangular band matrix_ vector_
 * multiplication.
 */
template <typename lhs_t, typename matrix_t, typename vector_t, typename sync_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE Trsv_2<
    lhs_t, matrix_t, vector_t, sync_t, local_range, is_upper, is_transposed,
    is_unitdiag>::Trsv_2(lhs_t &_l, matrix_t &_matrix,
                         typename Trsv_2<lhs_t, matrix_t, vector_t, sync_t,
                                         local_range, is_upper, is_transposed,
                                         is_unitdiag>::index_t &_blk_id,
                         vector_t &_vector, sync_t &_sync)
    : lhs_(_l),
      matrix_(_matrix),
      vector_(_vector),
      blk_id_(_blk_id),
      sync_(_sync) {}

template <typename lhs_t, typename matrix_t, typename vector_t, typename sync_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE typename Trsv_2<lhs_t, matrix_t, vector_t, sync_t, local_range,
                                 is_upper, is_transposed, is_unitdiag>::index_t
Trsv_2<lhs_t, matrix_t, vector_t, sync_t, local_range, is_upper, is_transposed,
       is_unitdiag>::get_size() const {
  return matrix_.get_size();
}
template <typename lhs_t, typename matrix_t, typename vector_t, typename sync_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE bool
Trsv_2<lhs_t, matrix_t, vector_t, sync_t, local_range, is_upper, is_transposed,
       is_unitdiag>::valid_thread(cl::sycl::nd_item<1> ndItem) const {
  // Valid threads are established by ::eval.
  return true;
}

template <typename lhs_t, typename matrix_t, typename vector_t, typename sync_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
template <typename local_memory_t>
SYCL_BLAS_INLINE typename Trsv_2<lhs_t, matrix_t, vector_t, sync_t, local_range,
                                 is_upper, is_transposed, is_unitdiag>::value_t
Trsv_2<lhs_t, matrix_t, vector_t, sync_t, local_range, is_upper, is_transposed,
       is_unitdiag>::eval(local_memory_t local_mem,
                          cl::sycl::nd_item<1> ndItem) {
  constexpr bool is_forward =
      (is_upper && is_transposed) || (!is_upper && !is_transposed);

  const index_t _N = lhs_.get_size();
  const index_t l_idx = ndItem.get_local_id(0);
  const index_t _idx = ndItem.get_local_id(0)%local_range;
  const index_t l_idy = ndItem.get_local_id(0)/local_range;

  const index_t warpnum = 4;
  const index_t warpchunck = local_range/warpnum;

  auto l_x = local_mem.localAcc;
  // it would be better a index_t but we can end up in a situation of 64bit
  // atomics
  auto a = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
      sync_.eval(0));

  index_t bb;
  if (!l_idx) l_x[0] = (is_forward) ? a++ : a--;
  ndItem.barrier();
  const index_t block_id = l_x[0]; //group_broadcast(ndItem.get_group(), bb);

  const index_t _offset = block_id * local_range;
  const index_t g_idx = _offset + _idx;

  // BEGIN - solve extra-diagonal block
  index_t current_block =
      is_forward ? 0 : ((_N + local_range - 1) / local_range) - 1;

  value_t v = 0;

  const index_t _OFF0 = local_range*(1 + warpchunck*l_idy) + _idx;

  {  
    index_t _off0 = _OFF0;
    const index_t _off1 = current_block * local_range + warpchunck * l_idy;
    auto A = matrix_.get_pointer() + matrix_.getSizeL()*(current_block * local_range + warpchunck * l_idy);
    auto x = local_mem.localAcc.get_pointer() + _OFF0;
     #pragma unroll 
    for (index_t i = 0; i < warpchunck; ++i)
      /*if (g_idx < _N)*/ {
        *x  = (is_transposed) ? matrix_.eval(_off1 + i, g_idx)
                                                : A[g_idx];
//        _off0 += local_range;
        x += local_range;
        A += matrix_.getSizeL(); 
      }
  } 

  const index_t i1 = warpchunck*l_idy;
  const index_t i2 = local_range*(1 + i1) + _idx; 
 
  volatile int *p = &sync_.eval(1);
  while (current_block != block_id) {
    //const index_t rbb = group_broadcast(ndItem.get_group(),*p);
    //const index_t rbb = *p; // little bit faster but unsafer
    if (!l_idx) l_x[0] = *p; 
    ndItem.barrier();
    const index_t rbb = l_x[0];

    while ((is_forward && (current_block < rbb)) ||
           (!is_forward && (current_block > rbb))) {
      const index_t _off = current_block * local_range;

      const index_t n_it = (_off + local_range < _N) ? local_range : _N - _off;

      if(l_idy == 0) l_x[_idx] = lhs_.eval(_off + _idx);

      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      
      auto l_x1 = local_mem.localAcc.get_pointer() + i1;
      auto l_x2 = local_mem.localAcc.get_pointer() + i2;
      #pragma unroll
      for (index_t i = 0; i < warpchunck; ++i){
        v += *l_x1 * *l_x2;
        l_x1++;
        l_x2+=local_range; 
      }

      if (is_forward)
        ++current_block;
      else
        --current_block;

      {  
  //      index_t _off0 = _OFF0; 
        const index_t _off1 = (current_block * local_range + warpchunck * l_idy);
        auto A = matrix_.get_pointer() + matrix_.getSizeL()*(current_block * local_range + warpchunck * l_idy);
        auto x = local_mem.localAcc.get_pointer() + _OFF0;
        #pragma unroll
        for (index_t i = 0; i < warpchunck; ++i)
        {
            /*l_x[_off0]*/ *x = (is_transposed) ? matrix_.eval(_off1 + i, g_idx)
                                                : A[g_idx];
            x += local_range;
  //          _off0 += local_range;
            A += matrix_.getSizeL(); 
        }
      } 

    }
   
  }
  // END - solve extra-diagonal block

  const index_t i3 = local_range*(1 + local_range) + _idx;
 
  l_x[i3 + local_range*l_idy] = v; 
  
  sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::work_group);

  if (l_idy == 0) {

  #pragma unroll
  for(index_t ii = 1; ii < warpnum; ++ii)
     v += l_x[i3 + local_range*ii];


  if (g_idx < _N) l_x[_idx] = lhs_.eval(g_idx) - v;


  // BEGIN - solve diagonal block
  value_t r_x = l_x[_idx];
  const index_t n_it =
      (_offset + local_range < _N) ? local_range : _N - _offset;

  auto A_diag_idx = local_mem.localAcc.get_pointer() + local_range + _idx;
  auto A_diag_diag = local_mem.localAcc.get_pointer() + local_range;

  #pragma unroll 32 
  for (index_t _it = 0; _it < local_range/*n_it*/; ++_it) {
    const index_t l_diag = (is_forward) ? _it : n_it - 1 - _it;
    
    if (!is_unitdiag && (l_idx == l_diag))
      r_x=sycl::native::divide(r_x, *A_diag_diag);

    value_t x_diag = sycl::group_broadcast(ndItem.get_sub_group(), r_x, l_diag);

    r_x -= *A_diag_idx * x_diag *
        ((((l_idx > l_diag) && /*(g_idx < _N) && +3ms*/ is_forward) ||
        ((l_idx < l_diag) && !is_forward)) ? value_t(1) : value_t(0));

    A_diag_idx+=local_range;
    A_diag_diag+=(local_range+1);
  }
  // END - solve diagonal block

  // Copy to memory the final result, this will be last in any case.
  if (g_idx < _N) lhs_.eval(g_idx) = r_x;

}

  sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);
  volatile int * sync = sync_.get_pointer() + 1;
  if (!l_idx) *sync = block_id+1;
  sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);

  return 0;
}

template <typename lhs_t, typename matrix_t, typename vector_t, typename sync_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE void
Trsv_2<lhs_t, matrix_t, vector_t, sync_t, local_range, is_upper, is_transposed,
       is_unitdiag>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  matrix_.bind(h);
  vector_.bind(h);
  sync_.bind(h);
}
template <typename lhs_t, typename matrix_t, typename vector_t, typename sync_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE void
Trsv_2<lhs_t, matrix_t, vector_t, sync_t, local_range, is_upper, is_transposed,
       is_unitdiag>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  matrix_.adjust_access_displacement();
  vector_.adjust_access_displacement();
  sync_.adjust_access_displacement();
}

}  // namespace blas
#endif
