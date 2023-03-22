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

  // is the not first wi of the wg 
  const bool is_not_wi0 = ndItem.get_local_id(0);

  // local bi-dimensional indexes 
  const index_t _idx = ndItem.get_local_id(0)%local_range;
  const index_t _idy = ndItem.get_local_id(0)/local_range;

  // number of warps per wg
  const index_t warpnum = 4; // <-- this will be a template argument
  const index_t warpchunck = local_range/warpnum;

  // pointers to local memory
  value_t * const loc_A = local_mem.localAcc.get_pointer();
  value_t * const tmp_A = local_mem.localAcc.get_pointer() + (local_range*warpchunck*_idy);
  value_t * const loc_x = local_mem.localAcc.get_pointer() + (local_range*local_range); 
  value_t * const tmp_x = loc_x + local_range + (local_range*_idy); 


  auto a = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
      sync_.eval(0));
  // get the block_id of actual workgroup
  const index_t block_id = group_broadcast(ndItem.get_group(), is_not_wi0 ? 0 : (is_forward) ? a++ : a--);


  // global memory offsets
  const index_t _offset = block_id * local_range;
  const index_t g_idx = _offset + _idx;

  // initialize current_block 
  index_t current_block = sycl::group_broadcast(ndItem.get_sub_group(), _idx ? 0 : ((is_forward) ? 0 : ((_N + local_range - 1) / local_range) - 1));


  // read first block of the row

  {  
    value_t * lA = tmp_A;
    value_t * gA = matrix_.get_pointer() + matrix_.getSizeL()*(current_block * local_range + warpchunck * _idy);
    #pragma unroll 
    for (index_t i = 0; i < warpchunck; ++i)
    {
        lA[_idx]  = ((current_block * local_range + warpchunck * _idy + i < _N) && (g_idx<_N)) ? gA[g_idx] : value_t(0);
        lA += local_range;
        gA += matrix_.getSizeL(); 
    }
  } 


  // initialize private accumulation value
  value_t v = 0;

  // BEGIN - solve extra-diagonal block

  volatile int *p = &sync_.eval(1);
  while (current_block != block_id) {
    const index_t rbb = group_broadcast(ndItem.get_group(), is_not_wi0 ? 0 : *p);
    while ((is_forward && (current_block < rbb)) ||
           (!is_forward && (current_block > rbb))) {
      const index_t _off = current_block * local_range;

      const index_t n_it = (_off + local_range < _N) ? local_range : _N - _off;
      if (_idy == 0) loc_x[_idx] = (g_idx<_N) ? lhs_.eval(_off + _idx) : value_t(0);

      ndItem.barrier(cl::sycl::access::fence_space::local_space);

      value_t * lx = loc_x + _idy * warpchunck;
      value_t * lA = tmp_A;
      #pragma unroll
      for (index_t i = 0; i < warpchunck; ++i){
        v += lA[_idx] * *lx;
        lx++;
        lA+=local_range; 
      }

      if (is_forward)
        ++current_block;
      else
        --current_block;

      {  
        value_t * lA = tmp_A;
        value_t * gA = matrix_.get_pointer() + matrix_.getSizeL()*(current_block * local_range + warpchunck * _idy);
        #pragma unroll 
        for (index_t i = 0; i < warpchunck; ++i)
        {
            lA[_idx]  = ((current_block * local_range + warpchunck * _idy + i < _N) && (g_idx<_N)) ? gA[g_idx] : value_t(0);
            lA += local_range;
            gA += matrix_.getSizeL(); 
        }
      } 

    }
   
  }
  // END - solve extra-diagonal block

  if (_idy != 0) tmp_x[_idx] = v; 
  
  //sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::work_group);
  ndItem.barrier(cl::sycl::access::fence_space::local_space);

  if (_idy == 0) {

  #pragma unroll
  for(index_t y = 1; y < warpnum; ++y)
     v += tmp_x[local_range*y + _idx];
  
 // BEGIN - solve diagonal block

  // compute recip (eventually move above)
  const value_t A_diag_recip = (g_idx < _N) ? sycl::native::recip(loc_A[local_range*_idx + _idx]) : value_t(0);
  value_t _A, r_diag, r_x;
  r_x = (g_idx < _N) ? lhs_.eval(g_idx) - v : value_t(0);

  #pragma unroll 
  for (index_t _it = 0; _it < local_range; ++_it) {
    const index_t l_diag = (is_forward) ? _it : local_range - 1 - _it;
    r_diag = sycl::group_broadcast(ndItem.get_sub_group(), is_unitdiag ? r_x : r_x*A_diag_recip, l_diag);
    _A = loc_A[local_range*l_diag + _idx];
    
    r_x -= _A* r_diag;

    if (_idx == l_diag) loc_x[_idx] = r_diag;

  }
  // END - solve diagonal block

  // Copy to memory the final result, this will be last in any case.
  if (g_idx < _N) lhs_.eval(g_idx) = loc_x[_idx];

}

  sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);
  volatile int * sync = sync_.get_pointer() + 1;
  if (!is_not_wi0) *sync = block_id+ ((is_forward) ? 1 : - 1);
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
