#ifndef TEMP_MEMORY_POOL_HPP
#define TEMP_MEMORY_POOL_HPP

#include "portblas_helper.h"

namespace blas {
#define VERBOSE
template <typename value_t>
typename helper::AllocHelper<value_t, helper::AllocType::buffer>::type
Temp_Mem_Pool::acquire_buff_mem(size_t size) {
  const size_t byteSize = size * sizeof(value_t);
  map_mutex_.lock();
  auto found = temp_buffer_map_.lower_bound(byteSize);
  if (found != temp_buffer_map_.end()) {
    cl::sycl::buffer<temp_buffer_map_t::mapped_type::value_type, 1> buff =
        found->second;
    temp_buffer_map_.extract(found);
    tot_size_temp_mem_ -= found->first;
    map_mutex_.unlock();
    return blas::BufferIterator<value_t>{buff.reinterpret<value_t>(
        cl::sycl::range<1>(found->first / sizeof(value_t)))};
  } else {
    map_mutex_.unlock();
#ifdef VERBOSE
    std::cout << "Create a temporary buffer of " << byteSize << " bytes."
              << std::endl;
#endif
    return make_sycl_iterator_buffer<value_t>(size);
  }
}

template <typename container_t>
cl::sycl::event Temp_Mem_Pool::release_buff_mem(
    std::vector<cl::sycl::event> dependencies, container_t mem) {
  return q_.submit([&, mem](cl::sycl::handler& cgh) {
    cgh.depends_on(dependencies);
    cgh.host_task([&, mem]() {
      const size_t byteSize = mem.get_buffer().byte_size();
      if (tot_size_temp_mem_ + byteSize <= max_size_temp_mem_) {
        map_mutex_.lock();
        tot_size_temp_mem_ += byteSize;
        temp_buffer_map_.emplace(
            byteSize,
            mem.get_buffer()
                .template reinterpret<
                    temp_buffer_map_t::mapped_type::value_type>(
                    cl::sycl::range<1>(
                        byteSize /
                        sizeof(temp_buffer_map_t::mapped_type::value_type))));
        map_mutex_.unlock();
      }
    });
  });
}

#ifdef SB_ENABLE_USM
template <typename value_t>
typename helper::AllocHelper<value_t, helper::AllocType::usm>::type
Temp_Mem_Pool::acquire_usm_mem(size_t size) {
  const size_t byteSize = size * sizeof(value_t);
  map_mutex_.lock();
  auto found = temp_usm_map_.lower_bound(byteSize);
  if (found != temp_usm_map_.end()) {
    temp_usm_map_.extract(found);
    tot_size_temp_mem_ -= found->first;
    map_mutex_.unlock();
    return reinterpret_cast<value_t*>(found->second);
  } else {
    map_mutex_.unlock();
#ifdef VERBOSE
    std::cout << "Create a temporary USM allocation of " << byteSize
              << " bytes." << std::endl;
#endif
    value_t* tmp = cl::sycl::malloc_device<value_t>(size, q_);
    map_mutex_.lock();
    temp_usm_size_map_.emplace(
        reinterpret_cast<temp_usm_size_map_t::key_type>(tmp), byteSize);
    map_mutex_.unlock();
    return tmp;
  }
}

template <typename container_t>
cl::sycl::event Temp_Mem_Pool::release_usm_mem(
    std::vector<cl::sycl::event> dependencies, container_t mem) {
  return q_.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(dependencies);
    cgh.host_task([&]() {
      map_mutex_.lock();
      auto found = temp_usm_size_map_.find(
          reinterpret_cast<temp_usm_size_map_t::key_type>(mem));
      const size_t byteSize = found->second;
      if (tot_size_temp_mem_ + byteSize > max_size_temp_mem_) {
        temp_usm_size_map_.erase(found);
        map_mutex_.unlock();
        cl::sycl::free(mem, q_);
      } else {
        tot_size_temp_mem_ += byteSize;
        temp_usm_map_.emplace(
            byteSize, reinterpret_cast<temp_usm_map_t::mapped_type>(mem));
        map_mutex_.unlock();
      }
    });
  });
}
}
#endif
#undef VERBOSE
#endif
