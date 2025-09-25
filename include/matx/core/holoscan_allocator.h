////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//  list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//  contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory_resource>
#include <memory>
#include "matx/core/allocator.h"
#include "matx/core/error.h"
#include "matx/core/storage.h"
#include "matx/core/tensor_desc.h"
#include "matx/core/nvtx.h"

#ifdef MATX_ENABLE_HOLOSCAN
#include <holoscan/core/resources/gxf/allocator.hpp>
#endif

namespace matx {

#ifdef MATX_ENABLE_HOLOSCAN

/**
 * @brief PMR-compatible memory resource that wraps Holoscan allocator
 * 
 * This class implements the C++17 std::pmr::memory_resource interface to provide
 * a bridge between Holoscan's memory allocation system and MatX's tensor operations.
 * It allows MatX tensors to use Holoscan's memory management while maintaining
 * compatibility with CUDA operations.
 * 
 * The memory resource integrates with MatX's memory tracking system to ensure
 * proper memory management and debugging capabilities.
 */
class holoscan_memory_resource : public std::pmr::memory_resource {
private:
  std::shared_ptr<holoscan::Allocator> holoscan_allocator_;
  matxMemorySpace_t memory_space_;

  /**
   * @brief Convert MatX memory space to Holoscan MemoryStorageType
   * 
   * @param space MatX memory space
   * @return Corresponding Holoscan MemoryStorageType
   */
  nvidia::gxf::MemoryStorageType matx_to_holoscan_memory_type(matxMemorySpace_t space) const {
    switch (space) {
      case MATX_DEVICE_MEMORY:
        return nvidia::gxf::MemoryStorageType::kDevice;
      case MATX_HOST_MEMORY:
        return nvidia::gxf::MemoryStorageType::kHost;
      case MATX_MANAGED_MEMORY:
        return nvidia::gxf::MemoryStorageType::kDevice;
      case MATX_HOST_PINNED_MEMORY:
        return nvidia::gxf::MemoryStorageType::kHost;
      default:
        return nvidia::gxf::MemoryStorageType::kDevice;
    }
  }

public:
  /**
   * @brief Construct a new holoscan memory resource object
   * 
   * @param allocator Shared pointer to Holoscan allocator
   * @param space Memory space for allocations (default: device memory)
   */
  holoscan_memory_resource(std::shared_ptr<holoscan::Allocator> allocator, 
                          matxMemorySpace_t space = MATX_DEVICE_MEMORY)
    : holoscan_allocator_(std::move(allocator)), memory_space_(space) {
    MATX_ASSERT_STR(holoscan_allocator_ != nullptr, matxInvalidParameter, 
                    "Holoscan allocator cannot be null");
  }

  /**
   * @brief Get the underlying Holoscan allocator
   * 
   * @return Shared pointer to the Holoscan allocator
   */
  std::shared_ptr<holoscan::Allocator> get_holoscan_allocator() const {
    return holoscan_allocator_;
  }

  /**
   * @brief Get the memory space used by this resource
   * 
   * @return MatX memory space
   */
  matxMemorySpace_t get_memory_space() const {
    return memory_space_;
  }

protected:
  /**
   * @brief Allocate memory using Holoscan allocator
   * 
   * @param bytes Number of bytes to allocate
   * @param alignment Memory alignment requirement
   * @return Pointer to allocated memory
   */
  void* do_allocate(size_t bytes, size_t alignment) override {
    MATX_ASSERT_STR(bytes > 0, matxInvalidParameter, "Allocation size must be greater than 0");
    
    auto storage_type = matx_to_holoscan_memory_type(memory_space_);
    
    // Track allocation in MatX's memory tracker
    void* ptr = nullptr;
    nvidia::byte* holoscan_ptr = holoscan_allocator_->allocate(static_cast<uint64_t>(bytes), storage_type);
    
    MATX_ASSERT_STR(holoscan_ptr != nullptr, matxMemoryError, "Holoscan allocation failed");
    
    ptr = static_cast<void*>(holoscan_ptr);
    
    // Check alignment requirements
    if (alignment > 1) {
      uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
      if (addr % alignment != 0) {
        holoscan_allocator_->free(holoscan_ptr);
        MATX_THROW(matxMemoryError, "Holoscan allocator returned unaligned memory");
      }
    }
    
    // Register with MatX memory tracker
    detail::GetMemoryTracker().Track(ptr, bytes, memory_space_);
    
    return ptr;
  }

  /**
   * @brief Deallocate memory using Holoscan allocator
   * 
   * @param ptr Pointer to memory to deallocate
   * @param bytes Size of the allocation (unused by Holoscan)
   * @param alignment Alignment of the allocation (unused by Holoscan)
   */
  void do_deallocate(void* ptr, size_t bytes, size_t alignment) override {
    if (ptr != nullptr) {
      // Untrack from MatX memory tracker
      detail::GetMemoryTracker().Untrack(ptr);
      
      // Free memory using Holoscan allocator
      holoscan_allocator_->free(static_cast<nvidia::byte*>(ptr));
    }
  }

  /**
   * @brief Check if this resource is equal to another
   * 
   * @param other Other memory resource to compare with
   * @return true if resources are equal, false otherwise
   */
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    const auto* other_holoscan = dynamic_cast<const holoscan_memory_resource*>(&other);
    if (other_holoscan == nullptr) {
      return false;
    }
    
    return holoscan_allocator_ == other_holoscan->holoscan_allocator_ &&
           memory_space_ == other_holoscan->memory_space_;
  }
};

/**
 * @brief MatX allocator that uses Holoscan memory resource
 * 
 * This allocator provides a MatX-compatible interface that uses a Holoscan
 * memory resource for actual memory allocation. It follows the same patterns
 * as the existing matx_allocator but delegates to Holoscan for memory management.
 * 
 * This allocator is designed to be used with raw_pointer_buffer to create
 * tensors that allocate memory through Holoscan's memory management system.
 * 
 * @tparam T Type of objects to allocate
 */
template <typename T>
class holoscan_allocator {
private:
  std::pmr::memory_resource* memory_resource_;

public:
  using value_type = T;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

  /**
   * @brief Construct a new holoscan allocator object
   * 
   * @param resource Pointer to PMR memory resource (must be holoscan_memory_resource)
   */
  explicit holoscan_allocator(std::pmr::memory_resource* resource = std::pmr::get_default_resource())
    : memory_resource_(resource) {
    MATX_ASSERT_STR(memory_resource_ != nullptr, matxInvalidParameter, 
                    "Memory resource cannot be null");
    
    // Verify that the resource is a holoscan_memory_resource
    auto* holoscan_resource = dynamic_cast<holoscan_memory_resource*>(memory_resource_);
    MATX_ASSERT_STR(holoscan_resource != nullptr, matxInvalidParameter,
                    "Memory resource must be a holoscan_memory_resource");
  }

  /**
   * @brief Copy constructor for different value types
   * 
   * @tparam U Other value type
   * @param other Other allocator to copy from
   */
  template <typename U>
  holoscan_allocator(const holoscan_allocator<U>& other) noexcept
    : memory_resource_(other.resource()) {
    // No need to verify resource type here as it was verified in the source allocator
  }

  /**
   * @brief Get the underlying memory resource
   * 
   * @return Pointer to the memory resource
   */
  std::pmr::memory_resource* resource() const noexcept {
    return memory_resource_;
  }

  /**
   * @brief Allocate memory for n objects of type T
   * 
   * @param n Number of objects to allocate space for
   * @return Pointer to allocated memory
   */
  T* allocate(size_t n) {
    if (n == 0) {
      return nullptr;
    }
    
    size_t bytes = n * sizeof(T);
    void* ptr = memory_resource_->allocate(bytes, alignof(T));
    
    // Memory tracking is handled in the memory_resource's do_allocate method
    return static_cast<T*>(ptr);
  }

  /**
   * @brief Deallocate memory for n objects of type T
   * 
   * @param ptr Pointer to memory to deallocate
   * @param n Number of objects (used for size calculation)
   */
  void deallocate(T* ptr, size_t n) {
    if (ptr != nullptr && n > 0) {
      size_t bytes = n * sizeof(T);
      // Memory untracking is handled in the memory_resource's do_deallocate method
      memory_resource_->deallocate(ptr, bytes, alignof(T));
    }
  }

  /**
   * @brief Deallocate memory using void pointer (MatX compatibility)
   * 
   * @param ptr Pointer to memory to deallocate
   * @param bytes Size of allocation in bytes
   */
  void deallocate(void* ptr, size_t bytes) {
    if (ptr != nullptr && bytes > 0) {
      // Memory untracking is handled in the memory_resource's do_deallocate method
      memory_resource_->deallocate(ptr, bytes, alignof(T));
    }
  }

  /**
   * @brief Check if two allocators are equal
   * 
   * @param other Other allocator to compare with
   * @return true if allocators are equal, false otherwise
   */
  bool operator==(const holoscan_allocator& other) const noexcept {
    return memory_resource_ == other.memory_resource_;
  }

  /**
   * @brief Check if two allocators are not equal
   * 
   * @param other Other allocator to compare with
   * @return true if allocators are not equal, false otherwise
   */
  bool operator!=(const holoscan_allocator& other) const noexcept {
    return !(*this == other);
  }
};

/**
 * @brief Create a tensor with Holoscan memory allocation using C array shape
 * 
 * @tparam T Element type of the tensor
 * @tparam RANK Rank (number of dimensions) of the tensor
 * @param shape Shape of tensor as C array
 * @param allocator Shared pointer to Holoscan allocator
 * @param space Memory space to allocate in
 * @param stream CUDA stream for allocation (currently unused with Holoscan)
 * @return New tensor using Holoscan memory allocation
 */
template <typename T, int RANK>
auto make_tensor_holoscan(const index_t (&shape)[RANK],
                         std::shared_ptr<holoscan::Allocator> allocator,
                         matxMemorySpace_t space = MATX_DEVICE_MEMORY,
                         cudaStream_t stream = 0) {
  MATX_NVTX_START("make_tensor_holoscan", matx::MATX_NVTX_LOG_API)

  // Create a memory resource and allocator
  auto memory_resource = std::make_shared<holoscan_memory_resource>(allocator, space);
  holoscan_allocator<T> alloc(memory_resource.get());
  
  // Create descriptor and calculate size
  DefaultDescriptor<RANK> desc{shape};
  size_t num_elements = static_cast<size_t>(desc.TotalSize());
  
  // Allocate memory using Holoscan allocator
  T* ptr = alloc.allocate(num_elements);
  
  // Create storage with ownership of the allocated memory
  raw_pointer_buffer<T, holoscan_allocator<T>> rp(ptr, num_elements * sizeof(T), true);
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T, RANK, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
}

/**
 * @brief Create a tensor with Holoscan memory allocation using container shape
 * 
 * @tparam T Element type of the tensor
 * @tparam ShapeType Type of shape container
 * @param shape Shape of tensor as container
 * @param allocator Shared pointer to Holoscan allocator
 * @param space Memory space to allocate in
 * @param stream CUDA stream for allocation (currently unused with Holoscan)
 * @return New tensor using Holoscan memory allocation
 */
template <typename T, typename ShapeType,
  std::enable_if_t<!is_matx_shape_v<ShapeType> &&
                   !is_matx_descriptor_v<ShapeType> &&
                   !std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
auto make_tensor_holoscan(ShapeType &&shape,
                         std::shared_ptr<holoscan::Allocator> allocator,
                         matxMemorySpace_t space = MATX_DEVICE_MEMORY,
                         cudaStream_t stream = 0) {
  MATX_NVTX_START("make_tensor_holoscan", matx::MATX_NVTX_LOG_API)

  // Create a memory resource and allocator
  auto memory_resource = std::make_shared<holoscan_memory_resource>(allocator, space);
  holoscan_allocator<T> alloc(memory_resource.get());
  
  // Create descriptor and calculate size
  constexpr int rank = static_cast<int>(cuda::std::tuple_size<typename remove_cvref<ShapeType>::type>::value);
  DefaultDescriptor<rank> desc{std::forward<ShapeType>(shape)};
  size_t num_elements = static_cast<size_t>(desc.TotalSize());
  
  // Allocate memory using Holoscan allocator
  T* ptr = alloc.allocate(num_elements);
  
  // Create storage with ownership of the allocated memory
  raw_pointer_buffer<T, holoscan_allocator<T>> rp(ptr, num_elements * sizeof(T), true);
  basic_storage<decltype(rp)> s{std::move(rp)};
  
  return tensor_t<T, rank, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
}

/**
 * @brief Create a 0D tensor with Holoscan memory allocation
 * 
 * @tparam T Element type of the tensor
 * @param t Unused empty initializer
 * @param allocator Shared pointer to Holoscan allocator
 * @param space Memory space to allocate in
 * @param stream CUDA stream for allocation (currently unused with Holoscan)
 * @return New 0D tensor using Holoscan memory allocation
 */
template <typename T>
auto make_tensor_holoscan([[maybe_unused]] const std::initializer_list<detail::no_size_t> t,
                         std::shared_ptr<holoscan::Allocator> allocator,
                         matxMemorySpace_t space = MATX_DEVICE_MEMORY,
                         cudaStream_t stream = 0) {
  MATX_NVTX_START("make_tensor_holoscan", matx::MATX_NVTX_LOG_API)
  
  cuda::std::array<index_t, 0> shape;
  return make_tensor_holoscan<T, decltype(shape)>(std::move(shape), allocator, space, stream);
}

#endif // MATX_ENABLE_HOLOSCAN

} // namespace matx
