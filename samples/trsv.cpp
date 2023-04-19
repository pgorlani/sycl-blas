#include "sycl_blas.hpp"
#include <CL/sycl.hpp>

#include "util.hpp"

int main(int argc, char** argv) {
  typedef int index_t;
  typedef float scalar_t;

  /* Create a SYCL queue with the default device selector */
  cl::sycl::queue q = cl::sycl::queue(cl::sycl::default_selector());
  auto myDev = q.get_device();
  auto myContext = q.get_context();
  auto myBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(myContext);

  /* Create a SYCL-BLAS sb_handle and get the policy handler */
  blas::SB_Handle sb_handle(q);

  auto kernelIds = sycl::get_kernel_ids();

  for (auto kernelId : kernelIds) {
    sycl::kernel myKernel = myBundle.get_kernel(kernelId);
    size_t maxSgSize =
        myKernel
            .get_info<sycl::info::kernel_device_specific::max_sub_group_size>(
                myDev);
    size_t compSgSize = myKernel.get_info<
        sycl::info::kernel_device_specific::compile_sub_group_size>(myDev);
    std::cerr << "sycl::info::kernel_device_specific::max_sub_group_size: "
              << maxSgSize
              << "\nsycl::info::kernel_device_specific::compile_sub_group_size: "
              << compSgSize << std::endl;
  }

  const bool trans = false;
  const bool is_upper = false;
  const bool is_unit = false;

  index_t incX = 1;
  index_t n = 31;
  index_t lda_mul = 1;

  const char* t_str = trans ? "t" : "n";
  const char* uplo_str = is_upper ? "u" : "l";
  const char* diag_str = is_unit ? "u" : "n";

  index_t a_size = n * n * lda_mul;
  index_t x_size = 1 + (n - 1) * incX;

  // Input matrix
  std::vector<scalar_t> a_m(a_size);
  // Input/output vector
  std::vector<scalar_t> x_v(x_size);

  // Control the magnitude of extra-diagonal elements
  for (index_t i = 0; i < n; ++i)
    for (index_t j = 0; j < n; ++j)
      a_m[(j * n * lda_mul) + i] =
          ((!is_upper && (i > j)) || (is_upper && (i < j))) ? scalar_t(1) : NAN;

  // Populate main diagonal with dominant elements
  for (index_t i = 0; i < n; ++i) {
    a_m[(i * n * lda_mul) + i] = scalar_t(1);
    x_v[i] = i + 1;
  }

  auto m_a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(a_m, a_size);
  auto v_x_gpu = blas::make_sycl_iterator_buffer<scalar_t>(x_v, x_size);

  _trsv(sb_handle, *uplo_str, *t_str, *diag_str, n, m_a_gpu, n * lda_mul,
        v_x_gpu, incX);

  auto event = blas::helper::copy_to_host(sb_handle.get_queue(), v_x_gpu,
                                          x_v.data(), x_size);
  sb_handle.wait(event);

  double maxerr = -1.0;
  for (index_t i = 0; i < x_size; i += incX) {
    maxerr = std::max(maxerr, std::fabs(double(x_v[i]) - double(1)));
    //    std::cerr<<" "<<x_v[i];
  }
  std::cerr << std::endl
            << " Maximum error compared to reference: " << maxerr << std::endl;

  return 0;
}
