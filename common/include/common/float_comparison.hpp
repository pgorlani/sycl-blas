/*
 * Copyright 2019 Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use these files except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * This file provides tools to compare floating-point numbers.
 * The function almost_equal returns a boolean indicating if two scalars can
 * be considered equal. The function compare_vectors checks if two vectors of
 * the same size are almost equal, prints a message on stderr if a value
 * mismatch is found, and returns a boolean as well.
 * Neither of the methods raises an exception nor calls exit(1)
 */

#ifndef UTILS_FLOAT_COMPARISON_H_
#define UTILS_FLOAT_COMPARISON_H_

#include <cmath>
#include <iostream>

#ifdef BLAS_DATA_TYPE_HALF
#if SYCL_LANGUAGE_VERSION < 202000
#include <CL/sycl.hpp>
inline std::ostream& operator<<(std::ostream& os, const cl::sycl::half& value) {
  os << static_cast<float>(value);
  return os;
}

namespace std {
template <>
class numeric_limits<cl::sycl::half> {
 public:
  static constexpr float min() { return -65504.0f; }
  static constexpr float max() { return 65504.0f; }
};
}  // namespace std
#endif  // SYCL_LANGUAGE_VERSION
#endif  // BLAS_DATA_TYPE_HALF

namespace utils {

template <typename scalar_t>
bool isnan(scalar_t value) noexcept {
  return std::isnan(value);
}

template <typename scalar_t>
bool isinf(scalar_t value) noexcept {
  return std::isinf(value);
}

template <typename scalar_t>
scalar_t abs(scalar_t value) noexcept {
  return std::abs(value);
}

#ifdef BLAS_DATA_TYPE_HALF
template <>
inline bool isnan<cl::sycl::half>(cl::sycl::half value) noexcept {
  return std::isnan(static_cast<float>(value));
}

template <>
inline bool isinf<cl::sycl::half>(cl::sycl::half value) noexcept {
  return std::isinf(static_cast<float>(value));
}

template <>
inline cl::sycl::half abs<cl::sycl::half>(cl::sycl::half value) noexcept {
  return std::abs(static_cast<float>(value));
}

#endif  // BLAS_DATA_TYPE_HALF

template <typename scalar_t>
scalar_t clamp_to_limits(scalar_t v) {
  constexpr auto min_value = std::numeric_limits<scalar_t>::min();
  constexpr auto max_value = std::numeric_limits<scalar_t>::max();
  if (decltype(min_value)(v) < min_value) {
    return min_value;
  } else if (decltype(max_value)(v) > max_value) {
    return max_value;
  } else {
    return v;
  }
}

/**
 * Indicates the tolerated margin for relative differences
 */
template <typename scalar_t>
inline scalar_t getRelativeErrorMargin() {
  /* Measured empirically with gemm. The dimensions of the matrices (even k)
   * don't seem to have an impact on the observed relative differences
   * In the cases where the relative error is relevant (non close to zero),
   * relative differences of up to 0.002 were observed for float
   */
  return static_cast<scalar_t>(0.005);
}

template <>
inline double getRelativeErrorMargin<double>() {
  /* Measured empirically with gemm. The dimensions of the matrices (even k)
   * don't seem to have an impact on the observed relative differences
   * In the cases where the relative error is relevant (non close to zero),
   * relative differences of up to 10^-12 were observed for double
   */
  return 0.0000000001;  // 10^-10
}

#ifdef BLAS_DATA_TYPE_HALF

template <>
inline cl::sycl::half getRelativeErrorMargin<cl::sycl::half>() {
  // Measured empirically with gemm
  return 0.05f;
}
#endif
/**
 * Indicates the tolerated margin for absolute differences (used in case the
 * scalars are close to 0)
 */
template <typename scalar_t>
inline scalar_t getAbsoluteErrorMargin() {
  /* Measured empirically with gemm.
   * In the cases where the relative error is irrelevant (close to zero),
   * absolute differences of up to 0.0006 were observed for float
   */
  return 0.001f;
}

template <>
inline double getAbsoluteErrorMargin<double>() {
  /* Measured empirically with gemm.
   * In the cases where the relative error is irrelevant (close to zero),
   * absolute differences of up to 10^-12 were observed for double
   */
  return 0.0000000001;  // 10^-10
}
#ifdef BLAS_DATA_TYPE_HALF

template <>
inline cl::sycl::half getAbsoluteErrorMargin<cl::sycl::half>() {
  // Measured empirically with gemm.
  return 1.0f;
}
#endif

/**
 * Compare two scalars and returns false if the difference is not acceptable.
 */
template <typename scalar_t, typename epsilon_t = scalar_t>
inline bool almost_equal(scalar_t const& scalar1, scalar_t const& scalar2) {
  // Shortcut, also handles case where both are zero
  if (scalar1 == scalar2) {
    return true;
  }
  // Handle cases where both values are NaN or inf
  if ((utils::isnan(scalar1) && utils::isnan(scalar2)) ||
      (utils::isinf(scalar1) && utils::isinf(scalar2))) {
    return true;
  }

  const scalar_t absolute_diff = utils::abs(scalar1 - scalar2);

  // Close to zero, the relative error doesn't work, use absolute error
  if (scalar1 == scalar_t{0} || scalar2 == scalar_t{0} ||
      absolute_diff < getAbsoluteErrorMargin<epsilon_t>()) {
    return (absolute_diff < getAbsoluteErrorMargin<epsilon_t>());
  }
  // Use relative error
  const auto absolute_sum = utils::abs(scalar1) + utils::abs(scalar2);
  return (absolute_diff / absolute_sum) < getRelativeErrorMargin<epsilon_t>();
}

template <typename FLOAT_TYPE>
class float_classifier {
 private:
  size_t _infs, _nans, _normals, _subnormals, _zeros;
  size_t _unknowns;
  size_t _total;

 public:
  float_classifier()
      : _infs(0),
        _nans(0),
        _normals(0),
        _subnormals(0),
        _zeros(0),
        _unknowns(0),
        _total(0){};

  void eval(FLOAT_TYPE f) {
    switch (std::fpclassify(f)) {
      case FP_INFINITE:
        ++_infs;
        break;
      case FP_NAN:
        ++_nans;
        break;
      case FP_NORMAL:
        ++_normals;
        break;
      case FP_SUBNORMAL:
        ++_subnormals;
        break;
      case FP_ZERO:
        ++_zeros;
        break;
      default:
        ++_unknowns;
    }
    ++_total;
  }

  size_t infs() const { return _infs; }
  size_t nans() const { return _nans; }
  size_t normals() const { return _normals; }
  size_t subnormals() const { return _subnormals; }
  size_t zeros() const { return _zeros; }
  size_t unknowns() const { return _unknowns; }
  size_t total() const { return _total; }
};

template <typename scalar_t>
inline void print_error_report(int inc, std::vector<scalar_t> const& vec,
                               std::vector<scalar_t> const& ref,
                               std::ostream& err_stream, std::string end_line);

/**
 * Compare two vectors and returns false if the difference is not acceptable.
 * The second vector is considered the reference.
 * @tparam scalar_t the type of data present in the input vectors
 * @tparam epilon_t the type used as tolerance. Lower precision types
 * (cl::sycl::half) will have a higher tolerance for errors
 */
template <typename scalar_t, typename epsilon_t = scalar_t>
inline bool compare_vectors(std::vector<scalar_t> const& vec,
                            std::vector<scalar_t> const& ref,
                            std::ostream& err_stream = std::cerr,
                            std::string end_line = "\n") {
  if (vec.size() != ref.size()) {
    err_stream << "Error: tried to compare vectors of different sizes"
               << std::endl;
    return false;
  }

  print_error_report(1, vec, ref, err_stream, end_line);

  for (int i = 0; i < vec.size(); ++i) {
    if (!almost_equal<scalar_t, epsilon_t>(vec[i], ref[i])) {
      err_stream << "Value mismatch at index " << i << ": " << vec[i]
                 << "; expected " << ref[i] << end_line;
      return false;
    }
  }
  return true;
}

/**
 * Compare two vectors at a given stride and window (unit_vec_size) and returns
 * false if the difference is not acceptable. The second vector is considered
 * the reference.
 * @tparam scalar_t the type of data present in the input vectors
 * @tparam epsilon_t the type used as tolerance. Lower precision types
 * (cl::sycl::half) will have a higher tolerance for errors
 * @param stride is the stride between two consecutive 'windows'
 * @param window is the size of a comparison window
 */
template <typename scalar_t, typename epsilon_t = scalar_t>
inline bool compare_vectors_strided(std::vector<scalar_t> const& vec,
                                    std::vector<scalar_t> const& ref,
                                    int stride, int window,
                                    std::ostream& err_stream = std::cerr,
                                    std::string end_line = "\n") {
  if (vec.size() != ref.size()) {
    err_stream << "Error: tried to compare vectors of different sizes"
               << std::endl;
    return false;
  }

  int k = 0;

  // Loop over windows
  while (window + (k + 1) * stride < vec.size()) {
    // Loop within a window
    for (int i = 0; i < window; ++i) {
      auto index = i + k * stride;
      if (!almost_equal<scalar_t, epsilon_t>(vec[index], ref[index])) {
        err_stream << "Value mismatch at index " << index << ": " << vec[index]
                   << "; expected " << ref[index] << end_line;
        return false;
      }
    }
    k += 1;
  }

  return true;
}

template <typename float_type>
inline uint64_t my_float_ulp_distance(float_type f_a, float_type f_b) {
  return 100;
}

template <>
inline uint64_t my_float_ulp_distance<float>(float f_a, float f_b) {
  static_assert(sizeof(float) == sizeof(uint32_t),
                "unsigned int/float sizes differ");

  uint32_t ulp;
  uint32_t a, b;

  std::memcpy(&a, &f_a, sizeof(float));
  std::memcpy(&b, &f_b, sizeof(float));

  if (a > b)
    ulp = a - b;
  else
    ulp = b - a;

  return ulp;
}

template <>
inline uint64_t my_float_ulp_distance<double>(double f_a, double f_b) {
  static_assert(sizeof(double) == sizeof(uint64_t),
                "unsigned int/float sizes differ");

  uint64_t ulp;
  uint64_t a, b;

  std::memcpy(&a, &f_a, sizeof(double));
  std::memcpy(&b, &f_b, sizeof(double));

  if (a > b)
    ulp = a - b;
  else
    ulp = b - a;

  return ulp;
}

/**
 */
template <typename scalar_t>
inline void print_error_report(int inc, std::vector<scalar_t> const& vec,
                               std::vector<scalar_t> const& ref,
                               std::ostream& err_stream, std::string end_line) {
  if (vec.size() != ref.size()) {
    err_stream << "Error: tried to compare vectors of different sizes"
               << std::endl;
  }

  float_classifier<scalar_t> _fpclass;

  long double max_abs_err = -1.0;
  long double max_rel_err = -1.0;
  uint64_t max_ulp_err = 0;
  long double tot_abs_err = 0;
  long double tot_rel_err = 0;
  long double tot_ulp_err = 0;

  for (int i = 0; i < vec.size(); i += inc) {
    _fpclass.eval(vec[i]);

    const long double delta =
        std::fabs((long double)vec[i] - (long double)ref[i]);
    const long double delta_rel = delta / std::fabs((long double)ref[i]);
    uint64_t delta_ulp = my_float_ulp_distance<scalar_t>(vec[i], ref[i]);

    max_abs_err = (max_abs_err > delta) ? max_abs_err : delta;
    max_rel_err = (max_rel_err > delta_rel) ? max_rel_err : delta_rel;
    max_ulp_err = (max_ulp_err > delta_ulp) ? max_ulp_err : delta_ulp;

    tot_abs_err += delta;
    tot_rel_err += delta_rel;
    tot_ulp_err += delta_ulp;
  }

  const double perc_tot = 100.0 / _fpclass.total();
  err_stream << "\nResult check" << end_line;
  err_stream << "Abs err (max)\t" << max_abs_err << "\t(mean)\t"
             << tot_abs_err / _fpclass.total() << end_line;
  err_stream << "Rel err (max)\t" << max_rel_err << "\t(mean)\t"
             << tot_rel_err / _fpclass.total() << end_line;
  err_stream << "ULP dis (max)\t" << max_ulp_err << "\t(mean)\t"
             << tot_ulp_err / _fpclass.total() << end_line;
  err_stream << "Fp-clss";
  err_stream << " #norm: " << _fpclass.normals() << " ("
             << perc_tot * _fpclass.normals() << " %)";
  err_stream << " #NaNs: " << _fpclass.nans() << " ("
             << perc_tot * _fpclass.nans() << " %)";
  err_stream << " #infs: " << _fpclass.infs() << " ("
             << perc_tot * _fpclass.infs() << " %)";
  err_stream << " #subn: " << _fpclass.subnormals() << " ("
             << perc_tot * _fpclass.subnormals() << " %)";
  err_stream << " #subn: " << _fpclass.unknowns() << " ("
             << perc_tot * _fpclass.unknowns() << " %)";
  err_stream << " #zero: " << _fpclass.zeros() << " ("
             << perc_tot * _fpclass.zeros() << " %)" << end_line;
}

}  // namespace utils

#endif  // UTILS_FLOAT_COMPARISON_H_
