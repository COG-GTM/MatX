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

#include "matx/core/nvtx.h"
#include "matx/core/storage.h"
#include "matx/core/tensor_desc.h"
#include "matx/core/allocator.h"

namespace matx {

/**
 * Create a tensor with a C array for the shape using Holoscan allocator
 *
 * @param shape Shape of tensor
 * @param space  memory space to allocate in.  Default is managed memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns New tensor
 **/
template <typename T, int RANK>
auto make_holoscan_tensor( const index_t (&shape)[RANK],
                          matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                          cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  T *ptr;
  DefaultDescriptor<RANK> desc{shape};

  size_t size = static_cast<size_t>(desc.TotalSize()) * sizeof(T);
  matxAlloc((void**)&ptr, size, space, stream);

  raw_pointer_buffer<T, holoscan_allocator<T>> rp(ptr, size, true);
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T, RANK, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
}

/**
 * Create a tensor from a conforming container type using Holoscan allocator
 *
 * Conforming containers have sequential iterators defined (both const and non-const). cuda::std::array
 * and std::vector meet this criteria.
 *
 * @param shape Shape of tensor
 * @param space  memory space to allocate in.  Default is managed memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns New tensor
 *
 **/
template <typename T, typename ShapeType,
  std::enable_if_t< !is_matx_shape_v<ShapeType> &&
                    !is_matx_descriptor_v<ShapeType> &&
                    !std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
auto make_holoscan_tensor( ShapeType &&shape,
                          matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                          cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  T *ptr;
  constexpr int rank = static_cast<int>(cuda::std::tuple_size<typename remove_cvref<ShapeType>::type>::value);
  DefaultDescriptor<rank> desc{std::move(shape)};

  size_t size = static_cast<size_t>(desc.TotalSize()) * sizeof(T);
  matxAlloc((void**)&ptr, size, space, stream);

  raw_pointer_buffer<T, holoscan_allocator<T>> rp(ptr, size, true);
  basic_storage<decltype(rp)> s{std::move(rp)};

  return tensor_t<T,
    cuda::std::tuple_size<typename remove_cvref<ShapeType>::type>::value,
    decltype(s),
    decltype(desc)>{std::move(s), std::move(desc)};
}

/**
 * Create a tensor with user-defined memory and a C array using Holoscan allocator
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 * @param owning
 *   If this class owns memory of data
 * @returns New tensor
 **/
template <typename T, int RANK>
auto make_holoscan_tensor( T *data,
                          const index_t (&shape)[RANK],
                          bool owning = false) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  DefaultDescriptor<RANK> desc{shape};
  raw_pointer_buffer<T, holoscan_allocator<T>> rp{data, static_cast<size_t>(desc.TotalSize())*sizeof(T), owning};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T, RANK, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
}

/**
 * Create a tensor with user-defined memory and conforming shape type using Holoscan allocator
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 * @param owning
 *    If this class owns memory of data
 * @returns New tensor
 **/
template <typename T, typename ShapeType,
  std::enable_if_t<!is_matx_descriptor_v<ShapeType> && !std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
auto make_holoscan_tensor( T *data,
                          ShapeType &&shape,
                          bool owning = false) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  constexpr int RANK = static_cast<int>(cuda::std::tuple_size<typename remove_cvref<ShapeType>::type>::value);
  DefaultDescriptor<RANK>
    desc{std::forward<ShapeType>(shape)};
  raw_pointer_buffer<T, holoscan_allocator<T>> rp{data, static_cast<size_t>(desc.TotalSize())*sizeof(T), owning};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T, RANK, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
}

} // namespace matx
