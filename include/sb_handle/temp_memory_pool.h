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
 *  @filename temp_memory_pool.h
 *
 **************************************************************************/
#ifndef TEMP_MEMORY_POOL_H
#define TEMP_MEMORY_POOL_H

#include <map>
#include <mutex>

namespace blas {
#define VERBOSE
class Temp_Mem_Pool {
 public:
  using queue_t = cl::sycl::queue;
  using event_t = std::vector<cl::sycl::event>;
  using temp_usm_map_t = std::multimap<size_t, void*>;
  using temp_usm_size_map_t = std::map<void*, size_t>;
  using temp_buffer_map_t = std::multimap<size_t, cl::sycl::buffer<int8_t, 1>>;

  Temp_Mem_Pool(queue_t q)
      : q_(q), tot_size_buff_mem_(0), tot_size_usm_mem_(0) {}
  Temp_Mem_Pool(const Temp_Mem_Pool& h) = delete;
  Temp_Mem_Pool operator=(Temp_Mem_Pool) = delete;

  ~Temp_Mem_Pool() {
    // wait for the completion of the host task
    q_.wait();

#ifdef VERBOSE
    std::cout << "Buffers destroyed on SB_Handle destruction: "
              << temp_buffer_map_.size() << std::endl;
#endif

#ifdef SB_ENABLE_USM
#ifdef VERBOSE
    std::cout << "USM allocations freed on SB_Handle destruction: "
              << temp_usm_map_.size() << std::endl;
#endif
    for (const temp_usm_map_t::value_type& p : temp_usm_map_)
      cl::sycl::free(p.second, q_);
#endif
  }

  inline queue_t get_queue() const { return q_; }

  template <typename value_t>
  typename helper::AllocHelper<value_t, helper::AllocType::buffer>::type
  acquire_buff_mem(size_t size);

  template <typename container_t>
  typename Temp_Mem_Pool::event_t release_buff_mem(
      const typename Temp_Mem_Pool::event_t&, const container_t&);

#ifdef SB_ENABLE_USM
  template <typename value_t>
  typename helper::AllocHelper<value_t, helper::AllocType::usm>::type
  acquire_usm_mem(size_t size);

  template <typename container_t>
  typename Temp_Mem_Pool::event_t release_usm_mem(
      const typename Temp_Mem_Pool::event_t&, const container_t&);
#endif

  temp_usm_map_t& temp_usm_map() { return temp_usm_map_; }
  temp_usm_size_map_t& temp_usm_size_map() { return temp_usm_size_map_; }
  temp_buffer_map_t& temp_buff_map() { return temp_buffer_map_; }

  template <typename container_t>
  void release_usm_mem_(const container_t& mem);

 private:
  static_assert(sizeof(temp_buffer_map_t::mapped_type::value_type) == 1);

  queue_t q_;
  size_t tot_size_buff_mem_;
  size_t tot_size_usm_mem_;
  static constexpr size_t max_size_temp_mem_ = 1e9;

  std::mutex map_mutex_;

  std::mutex temp_buffer_map_mutex_;
  temp_buffer_map_t temp_buffer_map_;
#ifdef SB_ENABLE_USM
  std::mutex temp_usm_map_mutex_;
  temp_usm_map_t temp_usm_map_;
  std::mutex temp_usm_size_map_mutex_;
  temp_usm_size_map_t temp_usm_size_map_;
#endif
};
#undef VERBOSE

}  // namespace blas
#endif
