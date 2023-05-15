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
 *  @filename blas2_tpsv_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename T>
using combination_t = std::tuple<index_t, bool, bool, bool, index_t, T>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  index_t n;
  bool trans;
  bool is_upper;
  bool is_unit;
  index_t incX;
  scalar_t unused; /* Work around dpcpp compiler bug
                      (https://github.com/intel/llvm/issues/7075) */
  std::tie(n, is_upper, trans, is_unit, incX, unused) = combi;

  const char* t_str = trans ? "t" : "n";
  const char* uplo_str = is_upper ? "u" : "l";
  const char* diag_str = is_unit ? "u" : "n";

  index_t a_size = ((n + 1) * n) / 2;
  index_t x_size = 1 + (n - 1) * incX;

  // Input matrix
  std::vector<scalar_t> a_m(a_size, 0);
  // Input/output vector
  std::vector<scalar_t> x_v(x_size, 1);
  // Input/output system vector
  std::vector<scalar_t> x_v_cpu(x_size);

  // Control the magnitude of extra-diagonal elements
  fill_random_with_range(a_m, scalar_t(-0.05), scalar_t(0.05));

  scalar_t* a_ptr = a_m.data();
  index_t stride = is_upper ? 2 : n;
  if (!is_unit) {
    // Populate main diagonal with dominant elements
    for (index_t i = 0; i < n; ++i) {
      *a_ptr = random_scalar(scalar_t(9), scalar_t(11));
      a_ptr += stride;
      is_upper ? stride++ : stride--;
    }
  }

  fill_random(x_v);

  /*
    for(int i = 0; i<x_size;++i)
      x_v[i] = i+1;
  */

  x_v_cpu = x_v;

  // SYSTEM TPSV
  reference_blas::tpsv(uplo_str, t_str, diag_str, n, a_m.data(), x_v_cpu.data(),
                       incX);

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);
  auto m_a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(a_m, a_size);
  auto v_x_gpu = blas::make_sycl_iterator_buffer<scalar_t>(x_v, x_size);

  // SYCL TPSV
  _tpsv(sb_handle, *uplo_str, *t_str, *diag_str, n, m_a_gpu, v_x_gpu, incX);

  auto event = blas::helper::copy_to_host(sb_handle.get_queue(), v_x_gpu,
                                          x_v.data(), x_size);
  sb_handle.wait(event);

#define PRINTMAXERR
#ifdef PRINTMAXERR
  double maxerr = -1.0;
  for (index_t i = 0; i < x_size; i += incX) {
    maxerr = std::max(maxerr, std::fabs(double(x_v[i]) - double(x_v_cpu[i])));
    //    std::cerr<<i<<": "<<x_v[i]<<" "<<x_v_cpu[i]<<std::endl;
  }
  std::cerr << std::endl
            << " Maximum error compared to reference: " << maxerr << std::endl;
#endif

  const bool isAlmostEqual = utils::compare_vectors(x_v, x_v_cpu);
  ASSERT_TRUE(isAlmostEqual);
}

#ifdef STRESS_TESTING
template <typename scalar_t>
const auto combi = ::testing::Combine(
    ::testing::Values(121, 288, 448, 553, 600, 996, 1024, 1999, 5252),  // n
    ::testing::Values(true, false),  // is_upper
    ::testing::Values(true, false),  // trans
    ::testing::Values(true, false),  // is_unit
    ::testing::Values(1, 2),         // incX
    ::testing::Values(0)             // unused
);
#else
// For the purpose of travis and other slower platforms, we need a faster test
// (the stress_test above takes about ~5 minutes)
template <typename scalar_t>
const auto combi = ::testing::Combine(
    ::testing::Values(16, 32, 48, 8192 /*14, 63, 257, 1010*/),  // n
    ::testing::Values(true, false),                             // is_upper
    ::testing::Values(true, false),                             // trans
    ::testing::Values(true, false),                             // is_unit
    ::testing::Values(1 /*, 2*/),                               // incX
    ::testing::Values(0)                                        // unused
);
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  index_t n, incX;
  bool is_upper;
  bool trans;
  bool is_unit;
  T unused;
  BLAS_GENERATE_NAME(info.param, n, is_upper, trans, is_unit, incX, unused);
}

BLAS_REGISTER_TEST_ALL(Tpsv, combination_t, combi, generate_name);
