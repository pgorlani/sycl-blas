#ifndef TEMP_MEMORY_POOL_HPP
#define TEMP_MEMORY_POOL_HPP

#include "portblas_helper.h"

namespace blas {
#define VERBOSE
template <typename value_t>
typename helper::AllocHelper<value_t, helper::AllocType::buffer>::type
Temp_Mem_Pool::acquire_buff_mem(size_t size) {
  const size_t byteSize = (size + size % 2) * sizeof(value_t);
  temp_buffer_map_mutex_.lock();
  auto found = temp_buffer_map_.lower_bound(byteSize);
  if (found != temp_buffer_map_.end()) {
    cl::sycl::buffer<temp_buffer_map_t::mapped_type::value_type, 1> buff =
        found->second;
    tot_size_buff_mem_ -= found->first;
    temp_buffer_map_.erase(found);
    temp_buffer_map_mutex_.unlock();
    return blas::BufferIterator<value_t>{buff.reinterpret<value_t>(
        cl::sycl::range<1>(buff.byte_size() / sizeof(value_t)))};
  } else {
    temp_buffer_map_mutex_.unlock();
#ifdef VERBOSE
    std::cout << "Create a temporary buffer of " << byteSize << " bytes."
              << std::endl;
#endif
    return make_sycl_iterator_buffer<value_t>(size);
  }
}

template <typename container_t>
typename Temp_Mem_Pool::event_t Temp_Mem_Pool::release_buff_mem(
    const typename Temp_Mem_Pool::event_t& dependencies,
    const container_t& mem) {
  return {q_.submit([&, mem](cl::sycl::handler& cgh) {
    cgh.depends_on(dependencies);
    cgh.host_task([&, mem]() {
      const size_t byteSize = mem.get_buffer().byte_size();
      if (tot_size_buff_mem_ + byteSize <= max_size_temp_mem_) {
        tot_size_buff_mem_ += byteSize;
        auto reinter_buff =
            mem.get_buffer()
                .template reinterpret<
                    temp_buffer_map_t::mapped_type::value_type>(
                    cl::sycl::range<1>(
                        byteSize /
                        sizeof(temp_buffer_map_t::mapped_type::value_type)));
        temp_buffer_map_mutex_.lock();
        temp_buffer_map_.emplace(byteSize, reinter_buff);
        temp_buffer_map_mutex_.unlock();
      }
    });
  })};
}

#ifdef SB_ENABLE_USM
template <typename value_t>
typename helper::AllocHelper<value_t, helper::AllocType::usm>::type
Temp_Mem_Pool::acquire_usm_mem(size_t size) {
  const size_t byteSize = size * sizeof(value_t);
  temp_usm_map_mutex_.lock();
  auto found = temp_usm_map_.lower_bound(byteSize);
  if (found != temp_usm_map_.end()) {
    tot_size_usm_mem_ -= found->first;
    value_t* tmp = reinterpret_cast<value_t*>(found->second);
    temp_usm_map_.erase(found);
    temp_usm_map_mutex_.unlock();
    return tmp;
  } else {
    temp_usm_map_mutex_.unlock();
#ifdef VERBOSE
    std::cout << "Create a temporary USM allocation of " << byteSize
              << " bytes." << std::endl;
#endif
    value_t* tmp = cl::sycl::malloc_device<value_t>(size, q_);
    temp_usm_size_map_mutex_.lock();
    temp_usm_size_map_.emplace(
        reinterpret_cast<temp_usm_size_map_t::key_type>(tmp), byteSize);
    temp_usm_size_map_mutex_.unlock();
    return tmp;
  }
}

template <typename container_t>
typename Temp_Mem_Pool::event_t Temp_Mem_Pool::release_usm_mem(
    const typename Temp_Mem_Pool::event_t& dependencies,
    const container_t& mem) {
  cl::sycl::context context = q_.get_context();
  return {q_.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(dependencies);
    cgh.host_task([&, mem, context]() { release_usm_mem_(mem); });
  })};
}

template <typename container_t>
void Temp_Mem_Pool::release_usm_mem_(const container_t& mem) {
  temp_usm_size_map_mutex_.lock();
  auto found = temp_usm_size_map().find(
      reinterpret_cast<temp_usm_size_map_t::key_type>(mem));
  const size_t byteSize = found->second;
  if (tot_size_usm_mem_ + byteSize > max_size_temp_mem_) {
    temp_usm_size_map_.extract(found);
    temp_usm_size_map_mutex_.unlock();
    cl::sycl::free(mem, q_);
  } else {
    temp_usm_size_map_mutex_.unlock();
    tot_size_usm_mem_ += byteSize;
    temp_usm_map_mutex_.lock();
    temp_usm_map_.emplace(byteSize,
                          reinterpret_cast<temp_usm_map_t::mapped_type>(mem));
    temp_usm_map_mutex_.unlock();
  }
}
}
#endif
#undef VERBOSE
#endif
