// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/builder/cuda/radix.h"
#include "data/eda_layout.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

void check(cudaError_t result)
{
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}

template<typename T>
struct DeviceBuffer {
  T *data = nullptr;
  explicit DeviceBuffer(size_t count) {
    check(cudaMalloc(reinterpret_cast<void **>(&data),count*sizeof(T)));
  }
  ~DeviceBuffer() { cudaFree(data); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

cudaError_t freeTreeBuffer(void *ptr)
{
#if CUDART_VERSION >= 11020
  return cudaFreeAsync(ptr,0);
#else
  return cudaFree(ptr);
#endif
}

template<typename T, int D>
struct BuiltTree {
  cuBQL::BinaryBVH<T,D> bvh;
  BuiltTree() = default;
  BuiltTree(const BuiltTree &) = delete;
  BuiltTree &operator=(const BuiltTree &) = delete;
  ~BuiltTree() noexcept {
    if (bvh.nodes) freeTreeBuffer(bvh.nodes);
    if (bvh.primIDs) freeTreeBuffer(bvh.primIDs);
  }
  void release() {
    if (bvh.nodes) { check(freeTreeBuffer(bvh.nodes)); bvh.nodes = nullptr; }
    if (bvh.primIDs) { check(freeTreeBuffer(bvh.primIDs)); bvh.primIDs = nullptr; }
  }
};

struct Event {
  cudaEvent_t handle;
  Event() { check(cudaEventCreate(&handle)); }
  ~Event() noexcept { cudaEventDestroy(handle); }
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
};

template<typename T, int D>
struct HostTree {
  std::vector<typename cuBQL::BinaryBVH<T,D>::node_t> nodes;
  std::vector<uint32_t> primIDs;
};

template<typename T, int D>
HostTree<T,D> snapshot(const cuBQL::BinaryBVH<T,D> &bvh)
{
  HostTree<T,D> result;
  result.nodes.resize(bvh.numNodes);
  result.primIDs.resize(bvh.numPrims);
  check(cudaMemcpy(result.nodes.data(),bvh.nodes,
                   result.nodes.size()*sizeof(result.nodes[0]),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(result.primIDs.data(),bvh.primIDs,
                   result.primIDs.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
  return result;
}

template<typename MortonKey>
__global__ void auditKeyBits(const MortonKey *keys, uint32_t count,
                             int bits, unsigned *flags)
{
  const uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= count) return;
  if ((keys[i] >> bits) != 0) atomicOr(flags,1u);
  if constexpr (sizeof(MortonKey) == 8)
    if ((keys[i] >> 32) != 0) atomicOr(flags,2u);
}

template<typename T, int D, typename MortonKey, int NumBits>
std::vector<MortonKey> auditGeneratedKeys(const cuBQL::box_t<T,D> *boxes,
                                          uint32_t count, bool requireHighBits=false)
{
  namespace radix = cuBQL::radixBuilder_impl;
  constexpr int bits = D*NumBits;
  static_assert(bits > 0 && bits < 8*sizeof(MortonKey));
  DeviceBuffer<radix::BuildState<T,D,MortonKey,NumBits>> state(1);
  DeviceBuffer<MortonKey> keys(count);
  DeviceBuffer<uint32_t> ids(count);
  DeviceBuffer<unsigned> flags(1);
  check(cudaMemset(flags.data,0,sizeof(unsigned)));
  radix::clearBuildState<T,D,MortonKey,NumBits><<<32,1>>>(state.data,count);
  radix::fillBuildState<T,D,MortonKey,NumBits><<<(count+1023)/1024,1024>>>(state.data,boxes,count);
  radix::finishBuildState<T,D,MortonKey,NumBits><<<32,1>>>(state.data);
  radix::computeUnsortedKeysAndPrimIDs<T,D,MortonKey,NumBits><<<(count+1023)/1024,1024>>>
    (keys.data,ids.data,state.data,boxes,count);
  auditKeyBits<<<(count+255)/256,256>>>(keys.data,count,bits,flags.data);
  check(cudaGetLastError());
  unsigned flagValue = 0;
  check(cudaMemcpy(&flagValue,flags.data,sizeof(flagValue),cudaMemcpyDeviceToHost));
  if (flagValue & 1u) throw std::runtime_error("Morton key exceeds populated bit range");
  if (requireHighBits && !(flagValue & 2u))
    throw std::runtime_error("64-bit Morton keys do not populate their upper 32 bits");
  std::vector<MortonKey> result(count);
  check(cudaMemcpy(result.data(),keys.data,count*sizeof(MortonKey),cudaMemcpyDeviceToHost));
  return result;
}

template<typename T, int D>
void validateTree(const HostTree<T,D> &tree, const std::vector<cuBQL::box_t<T,D>> &boxes)
{
  if (tree.nodes.empty() || tree.primIDs.size() != boxes.size())
    throw std::runtime_error("BVH size differs from the input");
  std::vector<unsigned char> seen(boxes.size(),0), covered(boxes.size(),0);
  for (uint32_t id : tree.primIDs)
    if (id >= boxes.size() || seen[id]++)
      throw std::runtime_error("primitive IDs are not a permutation of the input");
  std::vector<unsigned char> visited(tree.nodes.size(),0);
  std::vector<cuBQL::box_t<T,D>> bounds(tree.nodes.size());
  std::vector<std::pair<uint32_t,bool>> pending{{0,false}};
  size_t numVisited = 0;
  while (!pending.empty()) {
    const auto entry = pending.back(); pending.pop_back();
    const uint32_t id = entry.first;
    if (id >= tree.nodes.size()) throw std::runtime_error("BVH child exceeds node array");
    const auto &node = tree.nodes[id];
    if (!entry.second) {
      if (visited[id]++) throw std::runtime_error("BVH contains a cycle or shared child");
      ++numVisited;
      pending.emplace_back(id,true);
      if (node.admin.count) {
        const uint64_t end = uint64_t(node.admin.offset)+node.admin.count;
        if (end > tree.primIDs.size()) throw std::runtime_error("BVH leaf exceeds primitive array");
        for (uint64_t i=node.admin.offset;i<end;++i) {
          if (covered[size_t(i)]++) throw std::runtime_error("overlapping BVH leaf ranges");
          bounds[id].grow(boxes[tree.primIDs[size_t(i)]]);
        }
      } else {
        pending.emplace_back(uint32_t(node.admin.offset)+1,false);
        pending.emplace_back(uint32_t(node.admin.offset),false);
      }
    } else {
      if (!node.admin.count) {
        bounds[id].grow(bounds[uint32_t(node.admin.offset)]);
        bounds[id].grow(bounds[uint32_t(node.admin.offset)+1]);
      }
      for (int axis=0;axis<D;++axis)
        if (node.bounds.lower[axis] != bounds[id].lower[axis] ||
            node.bounds.upper[axis] != bounds[id].upper[axis])
          throw std::runtime_error("BVH bounds differ from descendant input bounds");
    }
  }
  if (numVisited+1 != tree.nodes.size() ||
      std::find(covered.begin(),covered.end(),0) != covered.end())
    throw std::runtime_error("unreachable BVH nodes or primitives");
}

template<typename T, int D>
void compareTrees(const HostTree<T,D> &a, const HostTree<T,D> &b, size_t count)
{
  if (a.nodes.size() != b.nodes.size() || a.primIDs != b.primIDs ||
      a.primIDs.size() != count)
    throw std::runtime_error("BVH size or sorted primitive IDs differ");
  std::vector<unsigned char> seen(count,0), covered(count,0);
  for (uint32_t id : a.primIDs) {
    if (id >= count || seen[id]++)
      throw std::runtime_error("primitive IDs are not a permutation of the input");
  }
  std::vector<std::pair<uint32_t,uint32_t>> pending{{0,0}};
  size_t visited = 0;
  while (!pending.empty()) {
    const auto pair = pending.back(); pending.pop_back();
    if (pair.first >= a.nodes.size() || pair.second >= b.nodes.size() ||
        ++visited > a.nodes.size())
      throw std::runtime_error("invalid BVH topology");
    const auto &left = a.nodes[pair.first];
    const auto &right = b.nodes[pair.second];
    if (left.admin.count != right.admin.count)
      throw std::runtime_error("leaf counts differ");
    for (int axis=0;axis<D;++axis) {
      const T lowerA = left.bounds.lower[axis], lowerB = right.bounds.lower[axis];
      const T upperA = left.bounds.upper[axis], upperB = right.bounds.upper[axis];
      if (std::memcmp(&lowerA,&lowerB,sizeof(T)) || std::memcmp(&upperA,&upperB,sizeof(T)))
        throw std::runtime_error("node bounds differ");
    }
    if (left.admin.count) {
      const uint64_t end = left.admin.offset+left.admin.count;
      if (left.admin.offset != right.admin.offset || end > count)
        throw std::runtime_error("leaf ranges differ or exceed primitive array");
      for (uint64_t i=left.admin.offset;i<end;++i)
        if (covered[size_t(i)]++) throw std::runtime_error("overlapping leaf ranges");
    } else {
      pending.emplace_back(uint32_t(left.admin.offset)+1,uint32_t(right.admin.offset)+1);
      pending.emplace_back(uint32_t(left.admin.offset),uint32_t(right.admin.offset));
    }
  }
  if (visited+1 != a.nodes.size() || std::find(covered.begin(),covered.end(),0) != covered.end())
    throw std::runtime_error("unreachable BVH nodes or primitives");
}

struct QueryResult { double area; uint32_t hits; unsigned overflow; };

__device__ bool positiveOverlap(const cuBQL::box_t<float,2> &a,
                                const cuBQL::box_t<float,2> &b)
{
  return a.lower[0] < b.upper[0] && b.lower[0] < a.upper[0] &&
         a.lower[1] < b.upper[1] && b.lower[1] < a.upper[1];
}

__global__ void queryAreas(cuBQL::BinaryBVH<float,2> bvh,
                          const cuBQL::box_t<float,2> *boxes,
                          const cuBQL::box_t<float,2> *queries,
                          QueryResult *results, uint32_t count)
{
  const uint32_t q = blockIdx.x*blockDim.x+threadIdx.x;
  if (q >= count) return;
  uint32_t stack[64], depth = 1;
  stack[0] = 0;
  QueryResult result{0,0,0};
  while (depth) {
    const auto node = bvh.nodes[stack[--depth]];
    if (!positiveOverlap(node.bounds,queries[q])) continue;
    if (!node.admin.count) {
      if (depth+2 > 64) { result.overflow = 1; break; }
      stack[depth++] = uint32_t(node.admin.offset)+1;
      stack[depth++] = uint32_t(node.admin.offset);
    } else {
      for (uint32_t i=0;i<node.admin.count;++i) {
        const auto box = boxes[bvh.primIDs[node.admin.offset+i]];
        if (!positiveOverlap(box,queries[q])) continue;
        const float dx = fminf(box.upper[0],queries[q].upper[0])-fmaxf(box.lower[0],queries[q].lower[0]);
        const float dy = fminf(box.upper[1],queries[q].upper[1])-fmaxf(box.lower[1],queries[q].lower[1]);
        result.area += double(dx)*double(dy);
        ++result.hits;
      }
    }
  }
  results[q] = result;
}

void auditQueries(const std::vector<cuBQL::box_t<float,2>> &boxes,
                  const cuBQL::box_t<float,2> *deviceBoxes,
                  const cuBQL::BinaryBVH<float,2> &a,
                  const cuBQL::BinaryBVH<float,2> &b,
                  const cuBQL::BinaryBVH<float,2> &c)
{
  using Box = cuBQL::box_t<float,2>;
  const eda_layout::Rectangle windows[] = {
    {-140,-140,-100,-90}, {-128,-128,0,0}, {-120,-116,-116,-92},
    {-16,-16,16,16}, {0,0,64,64}, {48,54,56,62}, {112,112,124,124},
    {-256,-256,-200,-200}, {-128,-128,896,384}, {100,4,104,40}
  };
  std::vector<Box> queries;
  for (const auto &r : windows) {
    Box box;
    box.lower[0] = float(r.x0); box.upper[0] = float(r.x1);
    box.lower[1] = float(r.y0); box.upper[1] = float(r.y1);
    queries.push_back(box);
  }
  DeviceBuffer<Box> deviceQueries(queries.size());
  DeviceBuffer<QueryResult> deviceResults(queries.size());
  check(cudaMemcpy(deviceQueries.data,queries.data(),queries.size()*sizeof(Box),cudaMemcpyHostToDevice));
  for (const auto &tree : {a,b,c}) {
    queryAreas<<<1,32>>>(tree,deviceBoxes,deviceQueries.data,deviceResults.data,uint32_t(queries.size()));
    check(cudaGetLastError());
    std::vector<QueryResult> results(queries.size());
    check(cudaMemcpy(results.data(),deviceResults.data,results.size()*sizeof(QueryResult),cudaMemcpyDeviceToHost));
    for (size_t q=0;q<queries.size();++q) {
      int64_t area = 0;
      uint32_t hits = 0;
      for (const auto &box : boxes) {
        const int64_t dx = std::min(int64_t(box.upper[0]),int64_t(queries[q].upper[0]))-
                           std::max(int64_t(box.lower[0]),int64_t(queries[q].lower[0]));
        const int64_t dy = std::min(int64_t(box.upper[1]),int64_t(queries[q].upper[1]))-
                           std::max(int64_t(box.lower[1]),int64_t(queries[q].lower[1]));
        if (dx > 0 && dy > 0) { area += dx*dy; ++hits; }
      }
      if (results[q].overflow || results[q].area != double(area) || results[q].hits != hits)
        throw std::runtime_error("2D query differs from independent CPU area/overlap oracle");
    }
  }
  std::cout << "query_oracle=PASS queries=" << queries.size() << " key_modes=3\n";
}

template<typename T>
void auditPrecisionQueries(const std::vector<cuBQL::box_t<T,2>> &boxes,
                           const cuBQL::box_t<T,2> *deviceBoxes,
                           const cuBQL::BinaryBVH<T,2> &a,
                           const cuBQL::BinaryBVH<T,2> &b,
                           const cuBQL::BinaryBVH<T,2> &c);

template<typename T, int D>
void auditBuilds(const std::vector<cuBQL::box_t<T,D>> &boxes,
                 const cuBQL::box_t<T,D> *deviceBoxes, bool pointQueries=false)
{
  namespace radix = cuBQL::radixBuilder_impl;
  constexpr int lowBits = radix::MortonKeyTraits<uint32_t,D>::numBits;
  constexpr int highBits = radix::MortonKeyTraits<uint64_t,D>::numBits;
  static_assert(lowBits == radix::numMortonBits<D>::value);
  const uint32_t count = uint32_t(boxes.size());
  const auto legacyKeys = auditGeneratedKeys<T,D,uint64_t,lowBits>(deviceBoxes,count);
  const auto keys32 = auditGeneratedKeys<T,D,uint32_t,lowBits>(deviceBoxes,count);
  auditGeneratedKeys<T,D,uint64_t,highBits>(deviceBoxes,count,true);
  if (!std::equal(legacyKeys.begin(),legacyKeys.end(),keys32.begin()))
    throw std::runtime_error("32-bit keys differ from legacy keys at equal quantization");
  BuiltTree<T,D> legacy, key32, key64;
  cuBQL::BuildConfig config(16);
  cuBQL::cuda::radixBuilder(legacy.bvh,deviceBoxes,count,config);
  cuBQL::cuda::radixBuilder<uint32_t>(key32.bvh,deviceBoxes,count,config);
  cuBQL::cuda::radixBuilder<uint64_t>(key64.bvh,deviceBoxes,count,config);
  check(cudaDeviceSynchronize());
  const auto legacyHost = snapshot(legacy.bvh);
  compareTrees(legacyHost,snapshot(key32.bvh),boxes.size());
  validateTree(legacyHost,boxes);
  validateTree(snapshot(key64.bvh),boxes);
  if constexpr (std::is_same<T,float>::value && D == 2)
    if (!pointQueries) auditQueries(boxes,deviceBoxes,legacy.bvh,key32.bvh,key64.bvh);
  if constexpr (D == 2)
    if (pointQueries) auditPrecisionQueries(boxes,deviceBoxes,legacy.bvh,key32.bvh,key64.bvh);
  legacy.release();
  key32.release();
  key64.release();
}

const char *patternName(eda_layout::Pattern pattern)
{
  return pattern == eda_layout::Pattern::Layout ? "layout" : "motif";
}

template<typename T, int D>
void runCase(const char *name, eda_layout::Pattern pattern)
{
  const auto boxes = eda_layout::makeBoxes<T,D>(8,4,pattern);
  DeviceBuffer<cuBQL::box_t<T,D>> deviceBoxes(boxes.size());
  check(cudaMemcpy(deviceBoxes.data,boxes.data(),boxes.size()*sizeof(boxes[0]),cudaMemcpyHostToDevice));
  auditBuilds(boxes,deviceBoxes.data);
  std::cout << "geometry=" << patternName(pattern) << " case=" << name
            << " boxes=" << boxes.size() << " geometry_fnv64="
            << std::hex << eda_layout::fingerprint(boxes) << std::dec
            << " key_modes=3 status=PASS\n";
}

void runCases(eda_layout::Pattern pattern)
{
  runCase<float,2>("float2",pattern); runCase<float,3>("float3",pattern); runCase<float,4>("float4",pattern);
  runCase<double,2>("double2",pattern); runCase<double,3>("double3",pattern); runCase<double,4>("double4",pattern);
  runCase<int,2>("int2",pattern); runCase<int,3>("int3",pattern); runCase<int,4>("int4",pattern);
  runCase<int64_t,2>("int64_2",pattern); runCase<int64_t,3>("int64_3",pattern); runCase<int64_t,4>("int64_4",pattern);
}

template<typename T, typename MortonKey, int NumBits>
void auditQuantizerBias(const cuBQL::box_t<T,2> *boxes, uint32_t count,
                        const cuBQL::vec_t<T,2> &expected)
{
  namespace radix = cuBQL::radixBuilder_impl;
  using State = radix::BuildState<T,2,MortonKey,NumBits>;
  DeviceBuffer<State> state(1);
  radix::clearBuildState<T,2,MortonKey,NumBits><<<1,1>>>(state.data,count);
  radix::fillBuildState<T,2,MortonKey,NumBits><<<1,1024>>>(state.data,boxes,count);
  radix::finishBuildState<T,2,MortonKey,NumBits><<<1,1>>>(state.data);
  check(cudaGetLastError());
  State hostState;
  check(cudaMemcpy(&hostState,state.data,sizeof(State),cudaMemcpyDeviceToHost));
  for (int axis=0;axis<2;++axis)
    if (hostState.quantizer.quantizeBias[axis] != expected[axis])
      throw std::runtime_error("quantizer centroid bounds lost input scalar precision");
}

template<typename T>
__global__ void queryPointHits(cuBQL::BinaryBVH<T,2> bvh,
                              const cuBQL::box_t<T,2> *boxes,
                              uint32_t *hits, unsigned *overflow, uint32_t count)
{
  const uint32_t q = blockIdx.x*blockDim.x+threadIdx.x;
  if (q >= count) return;
  const auto point = boxes[q].center();
  uint32_t stack[64], depth = 1, numHits = 0;
  stack[0] = 0;
  while (depth) {
    const auto node = bvh.nodes[stack[--depth]];
    if (point[0] < node.bounds.lower[0] || point[0] > node.bounds.upper[0] ||
        point[1] < node.bounds.lower[1] || point[1] > node.bounds.upper[1]) continue;
    if (node.admin.count) {
      for (uint32_t i=0;i<node.admin.count;++i) {
        const auto box = boxes[bvh.primIDs[node.admin.offset+i]];
        if (point[0] >= box.lower[0] && point[0] <= box.upper[0] &&
            point[1] >= box.lower[1] && point[1] <= box.upper[1]) ++numHits;
      }
    } else {
      if (depth+2 > 64) { atomicExch(overflow,1u); break; }
      stack[depth++] = uint32_t(node.admin.offset)+1;
      stack[depth++] = uint32_t(node.admin.offset);
    }
  }
  hits[q] = numHits;
}

template<typename T>
void auditPrecisionQueries(const std::vector<cuBQL::box_t<T,2>> &boxes,
                           const cuBQL::box_t<T,2> *deviceBoxes,
                           const cuBQL::BinaryBVH<T,2> &a,
                           const cuBQL::BinaryBVH<T,2> &b,
                           const cuBQL::BinaryBVH<T,2> &c)
{
  const uint32_t count = uint32_t(boxes.size());
  DeviceBuffer<uint32_t> hits(count);
  DeviceBuffer<unsigned> overflow(1);
  for (const auto &tree : {a,b,c}) {
    check(cudaMemset(overflow.data,0,sizeof(unsigned)));
    queryPointHits<<<1,256>>>(tree,deviceBoxes,hits.data,overflow.data,count);
    check(cudaGetLastError());
    unsigned overflowValue = 0;
    std::vector<uint32_t> actual(count);
    check(cudaMemcpy(&overflowValue,overflow.data,sizeof(unsigned),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual.data(),hits.data,count*sizeof(uint32_t),cudaMemcpyDeviceToHost));
    if (overflowValue) throw std::runtime_error("precision query exceeded traversal stack");
    for (uint32_t q=0;q<count;++q) {
      const auto point = boxes[q].center();
      uint32_t expected = 0;
      for (const auto &box : boxes)
        if (point[0] >= box.lower[0] && point[0] <= box.upper[0] &&
            point[1] >= box.lower[1] && point[1] <= box.upper[1]) ++expected;
      if (actual[q] != expected)
        throw std::runtime_error("precision query differs from independent CPU point oracle");
    }
  }
}

template<typename T>
void runPrecisionCase(const char *name, T origin, T span, T step,
                      bool requireFewerCollisions)
{
  namespace radix = cuBQL::radixBuilder_impl;
  constexpr int lowBits = radix::MortonKeyTraits<uint32_t,2>::numBits;
  constexpr int highBits = radix::MortonKeyTraits<uint64_t,2>::numBits;
  std::vector<cuBQL::box_t<T,2>> boxes(128), translated(128);
  for (size_t i=0;i<boxes.size();++i) {
    const T offset = i < 2 ? T(i)*span : span/T(2)+T(i-2)*step;
    for (int axis=0;axis<2;++axis) {
      boxes[i].lower[axis] = boxes[i].upper[axis] = origin+offset;
      translated[i].lower[axis] = translated[i].upper[axis] = offset;
    }
  }
  DeviceBuffer<cuBQL::box_t<T,2>> deviceBoxes(boxes.size()), deviceTranslated(boxes.size());
  check(cudaMemcpy(deviceBoxes.data,boxes.data(),boxes.size()*sizeof(boxes[0]),cudaMemcpyHostToDevice));
  check(cudaMemcpy(deviceTranslated.data,translated.data(),translated.size()*sizeof(translated[0]),cudaMemcpyHostToDevice));
  auditQuantizerBias<T,uint32_t,lowBits>(deviceBoxes.data,uint32_t(boxes.size()),boxes[0].center());
  auditQuantizerBias<T,uint64_t,highBits>(deviceBoxes.data,uint32_t(boxes.size()),boxes[0].center());
  auto keys32 = auditGeneratedKeys<T,2,uint32_t,lowBits>(deviceBoxes.data,uint32_t(boxes.size()));
  auto keys64 = auditGeneratedKeys<T,2,uint64_t,highBits>(deviceBoxes.data,uint32_t(boxes.size()),true);
  if (keys32 != auditGeneratedKeys<T,2,uint32_t,lowBits>(deviceTranslated.data,uint32_t(boxes.size())) ||
      keys64 != auditGeneratedKeys<T,2,uint64_t,highBits>(deviceTranslated.data,uint32_t(boxes.size()),true))
    throw std::runtime_error("Morton quantization changed after an exact coordinate translation");
  if (requireFewerCollisions) {
    std::sort(keys32.begin(),keys32.end()); std::sort(keys64.begin(),keys64.end());
    const auto unique32 = std::unique(keys32.begin(),keys32.end())-keys32.begin();
    const auto unique64 = std::unique(keys64.begin(),keys64.end())-keys64.begin();
    if (unique64 <= unique32)
      throw std::runtime_error("higher Morton precision did not resolve known low-precision collisions");
  }
  auditBuilds(boxes,deviceBoxes.data,true);
  std::cout << "precision_case=" << name << " boxes=" << boxes.size()
            << " key_modes=3 status=PASS\n";
}

float timedBuild(const cuBQL::box_t<float,2> *boxes, uint32_t count, bool key32)
{
  BuiltTree<float,2> tree;
  cuBQL::BuildConfig config(16);
  Event start, stop;
  check(cudaEventRecord(start.handle));
  if (key32)
    cuBQL::cuda::radixBuilder<uint32_t>(tree.bvh,boxes,count,config);
  else
    cuBQL::cuda::radixBuilder(tree.bvh,boxes,count,config);
  check(cudaGetLastError());
  check(cudaEventRecord(stop.handle)); check(cudaEventSynchronize(stop.handle));
  float ms = 0;
  check(cudaEventElapsedTime(&ms,start.handle,stop.handle));
  if (!std::isfinite(ms) || ms <= 0)
    throw std::runtime_error("CUDA event interval is not finite and positive");
  tree.release();
  return ms;
}

void benchmark(uint32_t tilesX, uint32_t tilesY, eda_layout::Pattern pattern)
{
  constexpr int meaningfulBits = cuBQL::radixBuilder_impl::MortonKeyTraits<uint32_t,2>::meaningfulBits;
  const auto boxes = eda_layout::makeBoxes<float,2>(tilesX,tilesY,pattern);
  cudaDeviceProp device{};
  int currentDevice = 0;
  check(cudaGetDevice(&currentDevice)); check(cudaGetDeviceProperties(&device,currentDevice));
  int runtimeVersion = 0, driverVersion = 0;
  check(cudaRuntimeGetVersion(&runtimeVersion)); check(cudaDriverGetVersion(&driverVersion));
  std::cout << "geometry=" << patternName(pattern) << " benchmark_device=" << device.name << "\n"
            << "compute_capability=" << device.major << '.' << device.minor
            << " device_bytes=" << device.totalGlobalMem
            << " cuda_runtime=" << runtimeVersion << " cuda_driver=" << driverVersion << "\n"
            << "tiles=" << tilesX << 'x' << tilesY << " boxes=" << boxes.size()
            << " input_bytes=" << boxes.size()*sizeof(boxes[0]) << " geometry_fnv64="
            << std::hex << eda_layout::fingerprint(boxes) << std::dec
            << " leaf=16 warmups_per_arm=1 balanced_pairs=6\n";
  DeviceBuffer<cuBQL::box_t<float,2>> deviceBoxes(boxes.size());
  check(cudaMemcpy(deviceBoxes.data,boxes.data(),boxes.size()*sizeof(boxes[0]),cudaMemcpyHostToDevice));
  auditBuilds(boxes,deviceBoxes.data);
  timedBuild(deviceBoxes.data,uint32_t(boxes.size()),false);
  timedBuild(deviceBoxes.data,uint32_t(boxes.size()),true);
  std::vector<float> times[2];
  for (int round=0;round<6;++round)
    for (int position=0;position<2;++position) {
      const int arm = (round+position)%2;
      const float ms = timedBuild(deviceBoxes.data,uint32_t(boxes.size()),arm != 0);
      times[arm].push_back(ms);
      std::cout << "round=" << round+1 << " position=" << position+1
                << " key_mode=" << (arm ? "uint32" : "legacy")
                << " key_bits=" << (arm ? 32 : 64)
                << " populated_bits=" << meaningfulBits
                << " end_bit=" << (arm ? meaningfulBits : 64) << " build_ms=" << ms << "\n";
    }
  for (auto &arm : times) std::sort(arm.begin(),arm.end());
  const float full = (times[0][2]+times[0][3])*0.5f;
  const float narrow = (times[1][2]+times[1][3])*0.5f;
  std::cout << "geometry=" << patternName(pattern)
            << " median_legacy64_ms=" << full << " median_key32_ms=" << narrow
            << " speedup=" << full/narrow << " validation=PASS\n";
}

uint32_t parseTileCount(const char *text, const char *name)
{
  const std::string error = std::string(name)+" must be a positive integer";
  uint32_t count = 0;
  if (!*text) throw std::invalid_argument(error);
  for (const char *p=text;*p;++p) {
    if (*p < '0' || *p > '9') throw std::invalid_argument(error);
    const uint32_t digit = uint32_t(*p-'0');
    if (count > (std::numeric_limits<uint32_t>::max()-digit)/10)
      throw std::invalid_argument(error);
    count = count*10+digit;
  }
  if (!count) throw std::invalid_argument(error);
  return count;
}

void validateBenchmarkSize(uint32_t tilesX, uint32_t tilesY)
{
  constexpr uint32_t maxTiles = (1u<<24)/eda_layout::tilePitch;
  if (tilesX > maxTiles || tilesY > maxTiles)
    throw std::invalid_argument("tile dimensions exceed exact float coordinate range");
  const uint64_t count = uint64_t(tilesX)*tilesY*eda_layout::rectanglesPerTile;
  if (count > uint64_t(std::numeric_limits<int>::max()/2))
    throw std::invalid_argument("box count exceeds radix builder integer range");
  if (count > std::numeric_limits<size_t>::max()/sizeof(cuBQL::box_t<float,2>))
    throw std::invalid_argument("input exceeds host address range");
}

void printHelp(const char *program)
{
  std::cout << "Usage: " << program << " [--help | --benchmark [tilesX tilesY]\n"
            << "       | --benchmark-motif [tilesX tilesY] | --write-layout output.svg]\n"
            << "  No arguments: check legacy, uint32_t, and higher-precision uint64_t keys for 12\n"
            << "    scalar/dimension cases on each pattern (24 cases, 1024 boxes each), plus three\n"
            << "    focused precision cases covering collisions and large double/int64_t origins.\n"
            << "  --benchmark: generate an original layout-inspired pattern; default 256x256 tiles\n"
            << "    produce 2097152 float2 boxes (32 MiB input). Use --benchmark 8 4 for a small run.\n"
            << "  --benchmark-motif: regenerate the historical A100 motif; default 1024x512 tiles\n"
            << "    produce 16777216 boxes (256 MiB input). Use --benchmark-motif 8 4 for a small run.\n"
            << "    Both benchmarks validate all three key modes, warm the legacy and uint32_t modes,\n"
            << "    then time six balanced pairs of complete radix-builder calls with CUDA events\n"
            << "    (leaf threshold 16).\n"
            << "    Timed modes use the same quantization: legacy uint64_t storage/full 64-bit sort\n"
            << "    versus actual uint32_t storage/populated-bit sort (30 bits in 2D). Explicit\n"
            << "    uint64_t keys use higher precision and are validated separately. No speedup\n"
            << "    threshold is required.\n"
            << "  --write-layout: write a 16x8 preview of the new layout pattern without initializing\n"
            << "    CUDA. Both geometries use deterministic integer coordinates generated in C++;\n"
            << "    no image input or external layout data is needed.\n";
}

} // namespace

int main(int argc, char **argv)
{
  try {
    std::cout << std::fixed << std::setprecision(6);
    if (argc == 2 && std::strcmp(argv[1],"--help") == 0) {
      printHelp(argv[0]);
    } else if (argc == 3 && std::strcmp(argv[1],"--write-layout") == 0) {
      eda_layout::writeSvg(argv[2],16,8);
      std::cout << "geometry=layout layout_svg=" << argv[2] << " tiles=16x8 boxes="
                << 16*8*eda_layout::rectanglesPerTile << "\n";
    } else if ((argc == 2 || argc == 4) &&
               (std::strcmp(argv[1],"--benchmark") == 0 || std::strcmp(argv[1],"--benchmark-motif") == 0)) {
      const bool motif = std::strcmp(argv[1],"--benchmark-motif") == 0;
      const uint32_t tilesX = argc == 4 ? parseTileCount(argv[2],"tilesX") : (motif ? 1024 : 256);
      const uint32_t tilesY = argc == 4 ? parseTileCount(argv[3],"tilesY") : (motif ? 512 : 256);
      validateBenchmarkSize(tilesX,tilesY);
      benchmark(tilesX,tilesY,motif ? eda_layout::Pattern::Motif : eda_layout::Pattern::Layout);
    } else if (argc == 1) {
      runCases(eda_layout::Pattern::Layout);
      runCases(eda_layout::Pattern::Motif);
      runPrecisionCase<float>("float2_collisions",0.f,float(1u<<20),1.f,true);
      runPrecisionCase<double>("double2_large_origin",double(uint64_t(1)<<40)+12345.,1024.,2.,false);
      runPrecisionCase<int64_t>("int64_2_large_origin",(int64_t(1)<<40)+12345,1024,2,false);
    } else {
      throw std::invalid_argument("invalid arguments; use --help for usage");
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Morton sort-range test failed: " << error.what() << '\n';
    return 1;
  }
}
