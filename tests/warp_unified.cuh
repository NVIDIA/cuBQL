// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Adapted from Andrea Paris' WarpUnified traversal (2a0f4db) and
// the lane-0 child-overlap change (96b9db49).

#include "cuBQL/bvh.h"
#include <cuda_runtime.h>

namespace warpUnified {

  using bvh2f = cuBQL::BinaryBVH<float,2>;
  using box2f = cuBQL::box_t<float,2>;
  using Admin = bvh2f::node_t::Admin;

  constexpr int warpSize = 32;
  constexpr int stackDepth = 32;
  constexpr unsigned fullWarp = 0xffffffffu;

  static_assert(sizeof(Admin) == sizeof(uint64_t),
                "WarpUnified requires an 8-byte Admin");

  inline __device__ Admin loadAdmin(const bvh2f::node_t *node)
  {
    Admin admin;
    admin.offsetAndCountBits = __ldg(&node->admin.offsetAndCountBits);
    return admin;
  }

  inline __device__ box2f loadBox(const box2f *box)
  {
    box2f result;
    result.lower.x = __ldg(&box->lower.x);
    result.lower.y = __ldg(&box->lower.y);
    result.upper.x = __ldg(&box->upper.x);
    result.upper.y = __ldg(&box->upper.y);
    return result;
  }

  inline __device__ void broadcastBox(box2f &box)
  {
    box.lower.x = __shfl_sync(fullWarp,box.lower.x,0);
    box.lower.y = __shfl_sync(fullWarp,box.lower.y,0);
    box.upper.x = __shfl_sync(fullWarp,box.upper.x,0);
    box.upper.y = __shfl_sync(fullWarp,box.upper.y,0);
  }

  inline __device__ bool overlaps(const box2f &query, const box2f &box)
  {
    return (query.lower.x <= box.upper.x)
      & (query.lower.y <= box.upper.y)
      & (box.lower.x <= query.upper.x)
      & (box.lower.y <= query.upper.y);
  }

  inline __device__ float intersectionArea(const box2f &query,
                                         const box2f &box)
  {
    const float x0 = fmaxf(query.lower.x,box.lower.x);
    const float y0 = fmaxf(query.lower.y,box.lower.y);
    const float x1 = fminf(query.upper.x,box.upper.x);
    const float y1 = fminf(query.upper.y,box.upper.y);
    const float dx = fmaxf(0.f,x1-x0);
    const float dy = fmaxf(0.f,y1-y0);
    return fmaf(dx,dy,0.f);
  }

  inline __device__ float accumulateLeafArea(const box2f &query,
                                           const box2f *reordered,
                                           uint32_t offset, uint32_t count)
  {
    const int lane = threadIdx.x & (warpSize-1);
    float area = 0.f;
    for (uint32_t i=lane;i<count;i+=warpSize)
      area += intersectionArea(query,loadBox(reordered+offset+i));
#pragma unroll
    for (int offset=warpSize/2;offset>0;offset>>=1)
      area += __shfl_down_sync(fullWarp,area,offset);
    return area;
  }

  struct TraversalState {
    int depth = 0;
    Admin admin{};
    bool active = true;
  };

  inline __device__ void pop(TraversalState &state, const Admin *stack)
  {
    if (state.depth > 0)
      state.admin = stack[--state.depth];
    else
      state.active = false;
  }

  inline __device__ void broadcastState(TraversalState &state)
  {
    state.admin.offsetAndCountBits
      = __shfl_sync(fullWarp,state.admin.offsetAndCountBits,0);
    state.depth = __shfl_sync(fullWarp,state.depth,0);
    state.active = __shfl_sync(fullWarp,int(state.active),0) != 0;
  }

  inline __device__ void descend(TraversalState &state, Admin *stack,
                                const bvh2f &bvh, uint32_t pairBase,
                                unsigned hits)
  {
    if (hits & 1u) {
      if (hits & 2u) {
        if (state.depth == stackDepth)
          asm volatile("trap;");
        stack[state.depth++] = loadAdmin(bvh.nodes+pairBase+1u);
      }
      state.admin = loadAdmin(bvh.nodes+pairBase);
    } else if (hits & 2u) {
      state.admin = loadAdmin(bvh.nodes+pairBase+1u);
    } else {
      state.admin.count = 0;
      pop(state,stack);
    }
  }

  // Called by a full warp with one query and a per-warp stack; lane 0 returns the area.
  template<bool Lane0>
  inline __device__ float queryArea(const bvh2f &bvh, box2f query,
                                   const box2f *reordered, Admin *stack)
  {
    if (bvh.numPrims == 0)
      return 0.f;

    const int lane = threadIdx.x & (warpSize-1);
    TraversalState state;
    if (lane == 0)
      state.admin = loadAdmin(bvh.nodes);
    broadcastState(state);

    float area = 0.f;
    while (state.active) {
      if (state.admin.count == 0) {
        const uint32_t pairBase = uint32_t(state.admin.offset);
        if constexpr (Lane0) {
          if (lane == 0) {
            const box2f bounds0 = loadBox(&bvh.nodes[pairBase].bounds);
            const box2f bounds1 = loadBox(&bvh.nodes[pairBase+1u].bounds);
            const unsigned hits = (overlaps(query,bounds0) ? 1u : 0u)
              | (overlaps(query,bounds1) ? 2u : 0u);
            descend(state,stack,bvh,pairBase,hits);
          }
        } else {
          box2f bounds0;
          box2f bounds1;
          if (lane == 0) {
            bounds0 = loadBox(&bvh.nodes[pairBase].bounds);
            bounds1 = loadBox(&bvh.nodes[pairBase+1u].bounds);
          }
          broadcastBox(bounds0);
          broadcastBox(bounds1);
          const unsigned hits = (overlaps(query,bounds0) ? 1u : 0u)
            | (overlaps(query,bounds1) ? 2u : 0u);
          if (lane == 0)
            descend(state,stack,bvh,pairBase,hits);
        }
        broadcastState(state);
      }

      if (state.active && state.admin.count != 0) {
        const float leafArea
          = accumulateLeafArea(query,reordered,uint32_t(state.admin.offset),
                               uint32_t(state.admin.count));
        if (lane == 0) {
          area += leafArea;
          pop(state,stack);
        }
        broadcastState(state);
      }
    }
    return area;
  }

  __global__ void gatherBoxesLeafOrderKernel(const box2f *src,
                                           const uint32_t *primIDs,
                                           int count, box2f *dst)
  {
    const uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < uint32_t(count))
      dst[i] = src[primIDs[i]];
  }

}
