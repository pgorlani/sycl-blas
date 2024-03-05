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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename ger.hpp
 *
 **************************************************************************/

#ifndef GER_HPP
#define GER_HPP

#include <operations/blas2_trees.h>
#include <operations/blas_operators.hpp>
#include <stdexcept>
#include <vector>
#include <views/view_sycl.hpp>

namespace blas {

/**** GER BY ROWS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
// template <typename lhs_t,  typename rhs_1_t, typename  rhs_2_t>
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
PORTBLAS_INLINE GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                        rhs_2_t>::GerRow(lhs_t &_l, value_t _scl, rhs_1_t &_r1,
                                         rhs_2_t &_r2, index_t &_nWG_row,
                                         index_t &_nWG_col,
                                         index_t &_shrMemSize)
    : lhs_(_l),
      scalar_(_scl),
      rhs_1_(_r1),
      rhs_2_(_r2),
      nWG_row_(_nWG_row),
      nWG_col_(_nWG_col),
      local_memory_size_(_shrMemSize) {}

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
PORTBLAS_INLINE typename GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                 rhs_2_t>::index_t
GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::get_size() const {
  return rhs_1_.get_size();
}
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
PORTBLAS_INLINE bool
GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
PORTBLAS_INLINE typename GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                 rhs_2_t>::value_t
GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::eval(
    typename GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                    rhs_2_t>::index_t i) {
  auto size =
      (lhs_.is_row_access()) ? lhs_.get_size_col() : lhs_.get_size_row();
  auto row = (lhs_.is_row_access()) ? (i / size) : (i % size);
  auto col = (lhs_.is_row_access()) ? (i % size) : (i / size);

  auto val = scalar_ * rhs_1_.eval(row) * rhs_2_.eval(col);

  return lhs_.eval(i) += val;
}

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
PORTBLAS_INLINE typename GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                 rhs_2_t>::value_t
GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  using index_t = typename GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                  rhs_2_t>::index_t;
  index_t localid = ndItem.get_local_id(0);
  index_t localSz = ndItem.get_local_range(0);
  index_t groupid = ndItem.get_group(0);

  index_t dimR = lhs_.get_size_row();
  index_t dimC = lhs_.get_size_col();

  index_t rowSz = (dimR + nWG_row_ - 1) / nWG_row_;

  index_t idWFR = (groupid % nWG_row_);
  index_t idWFC = (groupid / nWG_row_);
  index_t dimWFC =
      (dimC + (localSz * nWG_col_) - 1) / (localSz * nWG_col_) * localSz;

  index_t frs_row = idWFR * rowSz;
  index_t lst_row = std::min(dimR, frs_row + rowSz);

  index_t frs_col = idWFC * dimWFC + localid;
  index_t lst_col = std::min(dimC, frs_col + dimWFC);

  // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
  // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
  if ((!Upper && (((idWFC * dimWFC) + ((!Diag) ? 1 : 0)) > (lst_row - 1))) ||
      (!Lower &&
       ((frs_row + ((!Diag) ? 1 : 0)) > ((idWFC * dimWFC + dimWFC) - 1)))) {
    ;
  } else if (Single) {
    for (index_t colid = frs_col; colid < lst_col; colid += localSz) {
      auto val = scalar_ * rhs_2_.eval(colid);
      for (index_t id_row = frs_row, row = 0; id_row < lst_row;
           id_row++, row++) {
        if (Lower && Upper && Diag) {
          lhs_.eval(id_row, colid) += rhs_1_.eval(id_row) * val;
        } else {
          if ((Lower && ((colid + ((!Diag) ? 1 : 0)) <= id_row)) ||
              (Upper && (colid >= (id_row + ((!Diag) ? 1 : 0))))) {
            lhs_.eval(id_row, colid) += rhs_1_.eval(id_row) * val;
          }
        }
      }
    }
  } else {
    for (index_t colid = frs_col; colid < lst_col; colid += localSz) {
      auto val1 = scalar_ * rhs_1_.eval(colid);
      auto val2 = scalar_ * rhs_2_.eval(colid);
      for (index_t id_row = frs_row, row = 0; id_row < lst_row;
           id_row++, row++) {
        if (Lower && Upper && Diag) {
          lhs_.eval(id_row, colid) +=
              rhs_1_.eval(id_row) * val2 + val1 * rhs_2_.eval(id_row);
        } else {
          if ((Lower && ((colid + ((!Diag) ? 1 : 0)) <= id_row)) ||
              (Upper && (colid >= (id_row + ((!Diag) ? 1 : 0))))) {
            lhs_.eval(id_row, colid) +=
                rhs_1_.eval(id_row) * val2 + rhs_2_.eval(id_row) * val1;
          }
        }
      }
    }
  }

  return lhs_.eval(frs_row, frs_col);
}

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
template <typename sharedT>
PORTBLAS_INLINE typename GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                 rhs_2_t>::value_t
GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::eval(
    sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
  using index_t = typename GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                  rhs_2_t>::index_t;
  index_t localid = ndItem.get_local_id(0);
  index_t localSz = ndItem.get_local_range(0);
  index_t groupid = ndItem.get_group(0);

  index_t dimR = lhs_.get_size_row();
  index_t dimC = lhs_.get_size_col();

  index_t rowSz = (dimR + nWG_row_ - 1) / nWG_row_;
  index_t shrSz = local_memory_size_;

  index_t idWFR = (groupid % nWG_row_);
  index_t idWFC = (groupid / nWG_row_);
  index_t dimWFC =
      (dimC + (localSz * nWG_col_) - 1) / (localSz * nWG_col_) * localSz;

  index_t frs_row = idWFR * rowSz;
  index_t lst_row = std::min(dimR, frs_row + rowSz);

  index_t frs_col = idWFC * dimWFC + localid;
  index_t lst_col = std::min(dimC, frs_col + dimWFC);
  // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
  // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
  if ((!Upper && (((idWFC * dimWFC) + ((!Diag) ? 1 : 0)) > (lst_row - 1))) ||
      (!Lower &&
       ((frs_row + ((!Diag) ? 1 : 0)) > ((idWFC * dimWFC + dimWFC) - 1)))) {
    ;
  } else if (Single) {
    for (index_t rowid = frs_row; rowid < lst_row; rowid += shrSz) {
      if (rowid > frs_row)
        // This barrier is mandatory to be sure the data is on the shared
        // memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
      auto blqSz = std::min(shrSz, lst_row - rowid);
      for (index_t row = localid, id_row = rowid + localid; (row < blqSz);
           row += localSz, id_row += localSz) {
        shrMem[row] = scalar_ * rhs_1_.eval(id_row);
      }

      // This barrier is mandatory to be sure the data is on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);

      for (index_t colid = frs_col; (colid < lst_col); colid += localSz) {
        auto val = rhs_2_.eval(colid);
        for (index_t id_row = rowid, row = 0; row < blqSz; id_row++, row++) {
          if (Lower && Upper && Diag) {
            lhs_.eval(id_row, colid) += shrMem[row] * val;
          } else {
            if ((Lower && ((colid + ((!Diag) ? 1 : 0)) <= id_row)) ||
                (Upper && (colid >= (id_row + ((!Diag) ? 1 : 0))))) {
              lhs_.eval(id_row, colid) += shrMem[row] * val;
            }
          }
        }
      }
    }
  } else {
    auto shrSz1 = (shrSz >> 1);
    for (index_t rowid = frs_row; rowid < lst_row; rowid += shrSz) {
      if (rowid > frs_row)
        // This barrier is mandatory to be sure the data is on the shared
        // memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
      auto blqSz = std::min(shrSz1, lst_row - rowid);
      for (index_t row = localid, id_row = rowid + localid; (row < blqSz);
           row += localSz, id_row += localSz) {
        shrMem[row] = scalar_ * rhs_1_.eval(id_row);
        shrMem[shrSz1 + row] = scalar_ * rhs_2_.eval(id_row);
      }

      // This barrier is mandatory to be sure the data is on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);

      for (index_t colid = frs_col; (colid < lst_col); colid += localSz) {
        auto val1 = rhs_1_.eval(colid);
        auto val2 = rhs_2_.eval(colid);
        for (index_t id_row = rowid, row = 0; row < blqSz; id_row++, row++) {
          if (Lower && Upper && Diag) {
            lhs_.eval(id_row, colid) +=
                shrMem[row] * val2 + shrMem[shrSz1 + row] * val1;
          } else {
            if ((Lower && ((colid + ((!Diag) ? 1 : 0)) <= id_row)) ||
                (Upper && (colid >= (id_row + ((!Diag) ? 1 : 0))))) {
              lhs_.eval(id_row, colid) +=
                  shrMem[row] * val2 + shrMem[shrSz1 + row] * val1;
            }
          }
        }
      }
    }
  }

  return shrMem[0];
}
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
PORTBLAS_INLINE void GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                             rhs_2_t>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  rhs_1_.bind(h);
  rhs_2_.bind(h);
}
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
PORTBLAS_INLINE void GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                             rhs_2_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_1_.adjust_access_displacement();
  rhs_2_.adjust_access_displacement();
}

/**** GER BY COLUMNS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
// template <typename lhs_t,  typename rhs_1_t, typename  rhs_2_t>
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
PORTBLAS_INLINE GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                        rhs_2_t>::GerCol(lhs_t &_l, value_t _scl, rhs_1_t &_r1,
                                         rhs_2_t &_r2, index_t &_nWG_row,
                                         index_t &_nWG_col,
                                         index_t &_shrMemSize)
    : lhs_(_l),
      scalar_(_scl),
      rhs_1_(_r1),
      rhs_2_(_r2),
      nWG_row_(_nWG_row),
      nWG_col_(_nWG_col),
      local_memory_size_(_shrMemSize) {}

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
PORTBLAS_INLINE typename GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                 rhs_2_t>::index_t
GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::get_size() const {
  return rhs_1_.get_size();
}
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
PORTBLAS_INLINE bool
GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
PORTBLAS_INLINE typename GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                 rhs_2_t>::value_t
GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  using index_t = typename GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                  rhs_2_t>::index_t;

  const index_t group_id = ndItem.get_group(0);
  const index_t subgroup_id = ndItem.get_sub_group().get_group_id().get(0);
  const index_t subgroups_per_group= ndItem.get_sub_group().get_group_range().get(0);
  const index_t subgroup_size = ndItem.get_sub_group().get_local_range().get(0);
  const index_t subgroup_local_id = ndItem.get_sub_group().get_local_id().get(0);

  // Total size of the problem
  const index_t dimR = lhs_.get_size_row();
  const index_t dimC = lhs_.get_size_col();

  // Size of the block computed by a workgroup -- PARAMETERS
  const index_t block_rsize = 32;                // this must be equal to the sub-group size 
  const index_t block_csize = 32;                // 
  // CONSTRAIN col_chunck_size < subgroup_size
  const index_t subgroups_per_row = block_rsize/subgroup_size;
  const index_t col_chunck_size = block_csize/(subgroups_per_group/subgroups_per_row);

  // Block id's of the current workgroup
  const index_t idWFR = group_id % nWG_row_;
  const index_t idWFC = group_id / nWG_row_;

  // Compute the index offset for accessing data
  const index_t id_row0 = idWFR * block_rsize + subgroup_size * (subgroup_id%subgroups_per_row) +  subgroup_local_id;   //
  const index_t id_col0 = idWFC * block_csize + col_chunck_size * (subgroup_id/subgroups_per_row); //
  const bool id_row_active = id_row0 < dimR;

  //  
  const value_t rhs_2 = (subgroup_local_id < col_chunck_size && id_col0 + subgroup_local_id < dimC) ? rhs_2_.eval(id_col0 + subgroup_local_id) : 0;
  const value_t scal_rhs_1 = id_row_active ? scalar_ * rhs_1_.eval(id_row0) : 0; 
  value_t _lhs_ = (id_row_active && id_col0 < dimC) ? lhs_.eval(id_row0, id_col0) : 0;

  for (index_t sub_id_col = 0; sub_id_col < col_chunck_size; sub_id_col++) {
    const value_t rhs_2_sub_id_col = cl::sycl::group_broadcast(ndItem.get_sub_group(), rhs_2, sub_id_col);
    if(id_row_active && id_col0 + sub_id_col < dimC) {
      lhs_.eval(id_row0, id_col0 + sub_id_col) = _lhs_ + scal_rhs_1 * rhs_2_sub_id_col;
      _lhs_ = (id_col0 + sub_id_col + 1 < dimC) ? lhs_.eval(id_row0, id_col0 + sub_id_col + 1) : 0;
    }
  }

  return 0; 
}


template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
template <typename sharedT>
PORTBLAS_INLINE typename GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                 rhs_2_t>::value_t
GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::eval(
    sharedT shrMem, cl::sycl::nd_item<1> ndItem) {

  using index_t = typename GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                  rhs_2_t>::index_t;

  index_t group_id = ndItem.get_group(0);
  index_t group_size = ndItem.get_local_range(0);
  index_t group_local_id = ndItem.get_local_id(0);

  // Size of the block computed by a workgroup -- PARAMETERS
  const index_t block_rsize = 32;                // this must be equal to the sub-group size 
  const index_t block_csize = 32;                // 
 
  index_t idWFR = group_id % nWG_row_;
  index_t idWFC = group_id / nWG_row_;
  index_t frs_row = idWFR * block_rsize;
  const index_t id_row0 = group_local_id%block_rsize;  // CONSTRAIN group_size%block_rsize == 0
  const index_t id_row1 = frs_row + id_row0;
 
  index_t frs_col = idWFC * block_csize;

  const index_t dimR = lhs_.get_size_row();
  const index_t dimC = lhs_.get_size_col();

  value_t * l_rhs_1 = shrMem.localAcc.get_pointer();
  value_t * l_rhs_2 = shrMem.localAcc.get_pointer() + block_rsize;

  if (group_local_id < block_rsize)
    l_rhs_1[group_local_id] = (frs_row + group_local_id < dimR) ? scalar_ * rhs_1_.eval(frs_row + group_local_id) : 0; 

  if (group_local_id < block_csize)
    l_rhs_2[group_local_id] = (frs_col + group_local_id < dimC) ? rhs_2_.eval(frs_col + group_local_id) : 0;

  const index_t col_per_workitem = block_rsize * block_csize / group_size; // CONSTRAIN block_rsize * block_csize % group_size == 0
  const index_t chk_id = group_local_id/block_rsize;

  const index_t id_col0 = chk_id * col_per_workitem;
  const index_t id_col1 = frs_col + id_col0; 
 
  ndItem.barrier(cl::sycl::access::fence_space::local_space);

  for (index_t id_col = 0; id_col < col_per_workitem; id_col++)
  {
    const value_t val = l_rhs_1[id_row0] * l_rhs_2[id_col0 + id_col];
    if(id_row1 < dimR && id_col1 + id_col < dimC)
      lhs_.eval(id_row1, id_col1 + id_col) += val;
  }

  return 0;
}
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
PORTBLAS_INLINE void GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                             rhs_2_t>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  rhs_1_.bind(h);
  rhs_2_.bind(h);
}

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
PORTBLAS_INLINE void GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                             rhs_2_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_1_.adjust_access_displacement();
  rhs_2_.adjust_access_displacement();
}

}  // namespace blas

#endif
