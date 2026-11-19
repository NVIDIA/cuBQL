// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Builds BinaryBVHs with every GPU builder and checks each tree on the host:
// every valid prim is referenced by exactly one reachable leaf, no invalid
// (empty-box) prim is referenced, and every prim box and child node lies
// inside its parent's bounds.

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using box3 = cuBQL::box_t<float,3>;
using bvh3 = cuBQL::BinaryBVH<float,3>;

void check(cudaError_t result)
{
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}

bool inside(const box3 &in, const box3 &out)
{
  for (int d=0;d<3;d++)
    if (in.lower[d] < out.lower[d] || in.upper[d] > out.upper[d]) return false;
  return true;
}

bool valid(const box3 &b) { return b.lower[0] <= b.upper[0]; }

struct Errors { size_t missing = 0, duplicated = 0, invalid = 0, containment = 0; };

Errors checkTree(const bvh3 &bvh, const std::vector<box3> &boxes)
{
  std::vector<bvh3::Node> nodes(bvh.numNodes);
  std::vector<uint32_t> primIDs(bvh.numPrims);
  check(cudaMemcpy(nodes.data(),bvh.nodes,nodes.size()*sizeof(nodes[0]),
                   cudaMemcpyDeviceToHost));
  check(cudaMemcpy(primIDs.data(),bvh.primIDs,primIDs.size()*sizeof(primIDs[0]),
                   cudaMemcpyDeviceToHost));
  Errors e;
  std::vector<int> seen(boxes.size(),0);
  std::vector<uint32_t> stack = {0};
  while (!stack.empty()) {
    const auto &node = nodes[stack.back()];
    stack.pop_back();
    if (node.admin.count == 0) {
      for (int c=0;c<2;c++) {
        const uint32_t child = node.admin.offset+c;
        if (!inside(nodes[child].bounds,node.bounds)) e.containment++;
        stack.push_back(child);
      }
      continue;
    }
    for (uint32_t i=0;i<node.admin.count;i++) {
      const uint32_t primID = primIDs[node.admin.offset+i];
      if (primID >= boxes.size() || !valid(boxes[primID])) { e.invalid++; continue; }
      seen[primID]++;
      if (!inside(boxes[primID],node.bounds)) e.containment++;
    }
  }
  for (size_t i=0;i<boxes.size();i++) {
    if (!valid(boxes[i])) continue;
    e.missing    += seen[i] == 0;
    e.duplicated += seen[i] > 1;
  }
  return e;
}

enum Scene { UNIFORM, CLUSTERED, WITH_EMPTY };
const char *sceneName[] = { "uniform", "clustered", "with-empty" };

std::vector<box3> makeBoxes(size_t n, Scene scene, std::mt19937 &rng)
{
  std::uniform_real_distribution<float> u(0.f,1.f);
  std::vector<box3> boxes(n);
  for (size_t i=0;i<n;i++) {
    cuBQL::vec3f center;
    if (scene == CLUSTERED) {
      // 2000 clusters of near-coincident centers
      std::mt19937 clusterRng((unsigned)(i % 2000));
      center = cuBQL::vec3f(u(clusterRng),u(clusterRng),u(clusterRng))
        + cuBQL::vec3f(u(rng),u(rng),u(rng))*1e-7f;
    } else
      center = cuBQL::vec3f(u(rng),u(rng),u(rng));
    boxes[i].lower = center - cuBQL::vec3f(1e-3f);
    boxes[i].upper = center + cuBQL::vec3f(1e-3f);
    if (scene == WITH_EMPTY && i % 97 == 0)
      std::swap(boxes[i].lower,boxes[i].upper);
  }
  return boxes;
}

} // namespace

int main()
{
  const char *builderName[] = {
    "spatial median", "SAH", "ELH", "radix", "radix<uint64_t>", "rebinRadix"
  };
  std::mt19937 rng(42);
  int failures = 0;
  for (Scene scene : {UNIFORM, CLUSTERED, WITH_EMPTY})
    for (size_t n : {1000, 20000, 200000})
      for (int leafSize : {1, 4, 8}) {
        const std::vector<box3> boxes = makeBoxes(n,scene,rng);
        box3 *d_boxes = nullptr;
        check(cudaMalloc((void **)&d_boxes,n*sizeof(box3)));
        check(cudaMemcpy(d_boxes,boxes.data(),n*sizeof(box3),cudaMemcpyHostToDevice));
        // the radix builders require nonempty boxes
        const int numBuilders = scene == WITH_EMPTY ? 3 : 6;
        for (int b=0;b<numBuilders;b++) {
          bvh3 bvh;
          cuBQL::BuildConfig config(leafSize);
          if (b == 1) config.enableSAH();
          if (b == 2) config.enableELH();
          if (b <= 2)      cuBQL::gpuBuilder(bvh,d_boxes,(uint32_t)n,config);
          else if (b == 3) cuBQL::cuda::radixBuilder(bvh,d_boxes,(uint32_t)n,config);
          else if (b == 4) cuBQL::cuda::radixBuilder<uint64_t>(bvh,d_boxes,(uint32_t)n,config);
          else             cuBQL::cuda::rebinRadixBuilder(bvh,d_boxes,(uint32_t)n,config);
          check(cudaDeviceSynchronize());
          const Errors e = checkTree(bvh,boxes);
          const bool ok = !(e.missing || e.duplicated || e.invalid || e.containment);
          failures += !ok;
          printf("%-10s n=%-6zu leaf=%d %-15s %s (missing %zu, duplicated %zu, "
                 "invalid %zu, containment %zu)\n",
                 sceneName[scene],n,leafSize,builderName[b],ok ? "ok" : "FAILED",
                 e.missing,e.duplicated,e.invalid,e.containment);
          cuBQL::cuda::free(bvh);
        }
        check(cudaFree(d_boxes));
      }
  printf("%s\n",failures ? "FAILED" : "PASSED");
  return failures ? 1 : 0;
}
