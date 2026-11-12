// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/builder/cuda/radix.h"
#include "cuBQL/traversal/fixedAnyShapeQuery.h"
#include "data/eda_layout.h"
#include "warp_unified.cuh"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  using warpUnified::Admin;
  using warpUnified::box2f;
  using warpUnified::bvh2f;
  constexpr uint32_t blockSize = 64;
  constexpr uint32_t referenceBlockSize = 128;
  constexpr uint32_t referenceStackDepth = 64;
  constexpr uint32_t exactIntegerLimit = 1u<<24;
  constexpr int warmupCount = 2;
  constexpr int roundCount = 12;
  const char *variantNames[] = {"reference","before","after"};

  void check(cudaError_t error)
  {
    if (error != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(error));
  }

  template<typename T>
  struct DeviceBuffer {
    T *data = nullptr;

    explicit DeviceBuffer(size_t count)
    {
      if (count > std::numeric_limits<size_t>::max()/sizeof(T))
        throw std::runtime_error("device allocation size overflow");
      if (count)
        check(cudaMalloc(reinterpret_cast<void **>(&data),count*sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  };

  struct Event {
    cudaEvent_t handle = nullptr;
    Event() { check(cudaEventCreate(&handle)); }
    ~Event() { cudaEventDestroy(handle); }
    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;
  };

  struct Tree {
    bvh2f bvh;
    cuBQL::DeviceMemoryResource resource;
    ~Tree()
    {
      cudaFree(bvh.nodes);
      cudaFree(bvh.primIDs);
    }
    Tree() = default;
    Tree(const Tree &) = delete;
    Tree &operator=(const Tree &) = delete;
  };

  struct Options {
    uint32_t tilesX = 8, tilesY = 4;
    uint32_t queriesX = 0, queriesY = 0;
    uint32_t stride = 32, window = 100;
    bool benchmark = false, help = false;
  };

  struct Grid {
    uint32_t x, y, stride, window;
    int originX = -128, originY = -128;
  };

  struct QueryResult {
    uint64_t area = 0;
    uint32_t hits = 0;
  };

  uint32_t positiveInteger(const char *text)
  {
    errno = 0;
    char *end = nullptr;
    const auto value = std::strtoull(text,&end,10);
    if (!*text || *text == '-' || errno || *end || !value || value > INT_MAX)
      throw std::runtime_error("expected a positive integer <= INT_MAX");
    return uint32_t(value);
  }

  Options parseOptions(int argc, char **argv)
  {
    Options options;
    bool explicitTiles = false;
    bool explicitStride = false, explicitWindow = false;
    for (int i=1;i<argc;++i) {
      const std::string arg = argv[i];
      if (arg == "--help")
        options.help = true;
      else if (arg == "--benchmark")
        options.benchmark = true;
      else if ((arg == "--tiles" || arg == "--queries") && i+2 < argc) {
        const uint32_t x = positiveInteger(argv[++i]);
        const uint32_t y = positiveInteger(argv[++i]);
        if (arg == "--tiles") {
          options.tilesX = x;
          options.tilesY = y;
          explicitTiles = true;
        } else {
          options.queriesX = x;
          options.queriesY = y;
        }
      } else if (arg == "--stride" && i+1 < argc) {
        options.stride = positiveInteger(argv[++i]);
        explicitStride = true;
      } else if (arg == "--window" && i+1 < argc) {
        options.window = positiveInteger(argv[++i]);
        explicitWindow = true;
      } else
        throw std::runtime_error("unknown option or missing argument: "+arg);
    }
    if (options.help)
      return options;
    if (options.benchmark) {
      if (!explicitTiles) {
        options.tilesX = 256;
        options.tilesY = 256;
      }
      if (!explicitStride)
        options.stride = 216;
      if (!explicitWindow)
        options.window = 1792;
    }
    if (options.stride > exactIntegerLimit || options.window > exactIntegerLimit)
      throw std::runtime_error("query parameters exceed exact float-coordinate limits");
    if (options.tilesX > 131072 || options.tilesY > 131072)
      throw std::runtime_error("layout requires at most 131072 tiles per axis");
    const uint64_t boxCount = uint64_t(options.tilesX)*options.tilesY*eda_layout::rectanglesPerTile;
    if (boxCount > uint64_t(INT_MAX)/2
        || boxCount > std::numeric_limits<size_t>::max()/sizeof(box2f))
      throw std::runtime_error("layout exceeds builder node or allocation limits");
    if (!options.queriesX) {
      const uint64_t extentX = uint64_t(options.tilesX)*128;
      const uint64_t extentY = uint64_t(options.tilesY)*128;
      if (options.benchmark) {
        options.queriesX = uint32_t(1+(extentX > options.window ? extentX-options.window : 0)/options.stride);
        options.queriesY = uint32_t(1+(extentY > options.window ? extentY-options.window : 0)/options.stride);
      } else {
        options.queriesX = uint32_t((extentX+options.stride-1)/options.stride);
        options.queriesY = uint32_t((extentY+options.stride-1)/options.stride);
      }
    }
    const uint64_t count = uint64_t(options.queriesX)*options.queriesY;
    if (!count || count > INT_MAX
        || count > std::numeric_limits<size_t>::max()/sizeof(uint32_t))
      throw std::runtime_error("query grid exceeds indexing or allocation limits");
    const int64_t endX = -128+int64_t(options.queriesX-1)*options.stride+options.window;
    const int64_t endY = -128+int64_t(options.queriesY-1)*options.stride+options.window;
    if (endX > exactIntegerLimit || endY > exactIntegerLimit)
      throw std::runtime_error("query endpoints exceed exact float integer range");
    return options;
  }

  void printHelp()
  {
    std::cout << "Usage: cubql_warp_unified [--benchmark] [--tiles X Y] [--queries X Y]\n"
                 "                          [--stride N] [--window N]\n"
                 "Defaults: 8x4 tiles, stride 32, window 100, origin -128,-128.\n"
                 "--benchmark uses the Morton experiment's 256x256 layout (2097152 rectangles).\n"
                 "Benchmark defaults: 144x144 overlapping queries, stride 216, window 1792.\n"
                 "Compares scalar radix leaf1, Andrea WarpUnified, and the lane-0 fix.\n"
                 "The two WarpUnified variants share a radix leaf16 BVH and Morton query order.\n";
  }

  __host__ __device__ box2f queryBox(uint32_t id, Grid grid)
  {
    const int64_t x = int64_t(grid.originX)+int64_t(id%grid.x)*grid.stride;
    const int64_t y = int64_t(grid.originY)+int64_t(id/grid.x)*grid.stride;
    return box2f{cuBQL::vec2f{float(x),float(y)},
                 cuBQL::vec2f{float(x+grid.window),float(y+grid.window)}};
  }

  // Scalar baseline from window_test.cu at 396074e.
  template<typename Lambda>
  inline __cubql_both void forEachLeaf2D(const Lambda &lambda_to_call_on_each_leaf,
                                        const bvh2f bvh, const box2f query_box)
  {
    Admin traversal_stack[referenceStackDepth], *stack_ptr = traversal_stack;
    Admin node = bvh.nodes[0].admin;
    while (true) {
      while (true) {
        if (node.count != 0) {
          break;
        }
        const uint32_t n0_idx = static_cast<uint32_t>(node.offset)+0;
        const uint32_t n1_idx = static_cast<uint32_t>(node.offset)+1;
        const bvh2f::node_t n0 = bvh.nodes[n0_idx];
        const bvh2f::node_t n1 = bvh.nodes[n1_idx];
        const bool o0 = query_box.overlaps(n0.bounds);
        const bool o1 = query_box.overlaps(n1.bounds);
        if (o0) {
          if (o1) {
            *stack_ptr++ = n1.admin;
          }
          node = n0.admin;
        } else {
          if (o1) {
            node = n1.admin;
          } else {
            node.count = 0;
            break;
          }
        }
      }
      if (node.count != 0) {
        const int leaf_result = lambda_to_call_on_each_leaf(
          bvh.primIDs+node.offset,static_cast<uint32_t>(node.count));
        if (leaf_result == CUBQL_TERMINATE_TRAVERSAL) {
          return;
        }
      }
      if (stack_ptr == traversal_stack) {
        return;
      }
      node = *--stack_ptr;
    }
  }

  template<typename Lambda>
  inline __cubql_both void forEachPrim2D(const Lambda &lambda_to_call_on_each_prim,
                                        const bvh2f bvh, const box2f query_box)
  {
    auto leaf_code = [&lambda_to_call_on_each_prim]
      (const uint32_t *prim_ids, size_t num_prims) -> int {
        for (int i=0;i<static_cast<int>(num_prims);++i) {
          if (lambda_to_call_on_each_prim(prim_ids[i]) == CUBQL_TERMINATE_TRAVERSAL) {
            return CUBQL_TERMINATE_TRAVERSAL;
          }
        }
        return CUBQL_CONTINUE_TRAVERSAL;
      };
    forEachLeaf2D(leaf_code,bvh,query_box);
  }

  struct WindowAreaAccumulator {
    const box2f *boxes = nullptr;
    box2f window;
    mutable float total_area = 0.f;

    __device__ int operator()(uint32_t box_id) const
    {
      const box2f box = boxes[box_id];
      const float x0 = fmaxf(window.lower.x,box.lower.x);
      const float y0 = fmaxf(window.lower.y,box.lower.y);
      const float x1 = fminf(window.upper.x,box.upper.x);
      const float y1 = fminf(window.upper.y,box.upper.y);
      const float dx = fmaxf(0.f,x1-x0);
      const float dy = fmaxf(0.f,y1-y0);
      total_area += dx*dy;
      return CUBQL_CONTINUE_TRAVERSAL;
    }
  };

  __global__ void referenceQueryKernel(bvh2f bvh, const box2f *boxes, Grid grid,
                                      uint32_t count, float *results)
  {
    const uint64_t id = uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (id >= count)
      return;
    if (!bvh.numPrims) {
      results[id] = 0.f;
      return;
    }
    WindowAreaAccumulator accumulator{boxes,queryBox(uint32_t(id),grid),0.f};
    forEachPrim2D(accumulator,bvh,accumulator.window);
    results[id] = accumulator.total_area;
  }

  template<bool Lane0>
  __global__ void queryKernel(bvh2f bvh, const box2f *boxes, Grid grid,
                             const uint32_t *dispatch, uint32_t count, float *results)
  {
    const uint64_t warp = (uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)/warpUnified::warpSize;
    if (warp >= count)
      return;
    const uint32_t id = __ldg(&dispatch[warp]);
    Admin stack[warpUnified::stackDepth];
    const float area = warpUnified::queryArea<Lane0>(bvh,queryBox(id,grid),boxes,stack);
    if ((threadIdx.x&(warpUnified::warpSize-1)) == 0)
      results[id] = area;
  }

  void appendMorton(std::vector<uint32_t> &ids, uint32_t x, uint32_t y,
                    uint32_t side, Grid grid)
  {
    if (x >= grid.x || y >= grid.y)
      return;
    if (side == 1) {
      ids.push_back(y*grid.x+x);
      return;
    }
    const uint32_t half = side/2;
    appendMorton(ids,x,y,half,grid);
    appendMorton(ids,x+half,y,half,grid);
    appendMorton(ids,x,y+half,half,grid);
    appendMorton(ids,x+half,y+half,half,grid);
  }

  std::vector<uint32_t> makeDispatch(Grid grid)
  {
    const uint32_t count = grid.x*grid.y;
    std::vector<uint32_t> ids;
    ids.reserve(count);
    uint32_t side = 1;
    while (side < std::max(grid.x,grid.y))
      side *= 2;
    appendMorton(ids,0,0,side,grid);
    if (ids.size() != count)
      throw std::runtime_error("Morton dispatch misses queries");
    std::vector<unsigned char> seen(count,0);
    for (const uint32_t id : ids)
      if (id >= count || seen[id]++)
        throw std::runtime_error("Morton dispatch is not a permutation");
    return ids;
  }

  void launch(int variant, bvh2f reference, bvh2f optimized, const box2f *boxes,
              const box2f *reordered, Grid grid, const uint32_t *dispatch, float *results)
  {
    const uint32_t count = grid.x*grid.y;
    const uint32_t blocks = (count+1)/2;
    if (variant == 0)
      referenceQueryKernel<<<uint32_t((uint64_t(count)+referenceBlockSize-1)/referenceBlockSize),referenceBlockSize>>>(
        reference,boxes,grid,count,results);
    else if (variant == 1)
      queryKernel<false><<<blocks,blockSize>>>(optimized,reordered,grid,dispatch,count,results);
    else
      queryKernel<true><<<blocks,blockSize>>>(optimized,reordered,grid,dispatch,count,results);
    check(cudaGetLastError());
  }

  bool contains(const box2f &outer, const box2f &inner)
  {
    return outer.lower.x <= inner.lower.x && outer.lower.y <= inner.lower.y
      && outer.upper.x >= inner.upper.x && outer.upper.y >= inner.upper.y;
  }

  bool validBounds(const box2f &box)
  {
    return std::isfinite(box.lower.x) && std::isfinite(box.lower.y)
      && std::isfinite(box.upper.x) && std::isfinite(box.upper.y)
      && box.lower.x <= box.upper.x && box.lower.y <= box.upper.y;
  }

  void auditTree(bvh2f bvh, const std::vector<box2f> &boxes, const box2f *reordered,
                 uint32_t stackCapacity)
  {
    if (!bvh.nodes || !bvh.primIDs || !bvh.numNodes || bvh.numPrims != boxes.size())
      throw std::runtime_error("unexpected BVH counts or storage");
    std::vector<bvh2f::node_t> nodes(bvh.numNodes);
    std::vector<uint32_t> ids(bvh.numPrims);
    std::vector<box2f> gathered(reordered ? boxes.size() : 0);
    check(cudaMemcpy(nodes.data(),bvh.nodes,nodes.size()*sizeof(nodes[0]),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(ids.data(),bvh.primIDs,ids.size()*sizeof(ids[0]),cudaMemcpyDeviceToHost));
    if (reordered)
      check(cudaMemcpy(gathered.data(),reordered,gathered.size()*sizeof(gathered[0]),cudaMemcpyDeviceToHost));
    std::vector<unsigned char> primitiveSeen(boxes.size(),0), coverage(boxes.size(),0);
    for (size_t i=0;i<ids.size();++i) {
      if (ids[i] >= boxes.size() || primitiveSeen[ids[i]]++)
        throw std::runtime_error("BVH primitive IDs are not a permutation");
      if (reordered) {
        const auto &expected = boxes[ids[i]];
        const auto &actual = gathered[i];
        if (actual.lower.x != expected.lower.x || actual.lower.y != expected.lower.y
            || actual.upper.x != expected.upper.x || actual.upper.y != expected.upper.y)
          throw std::runtime_error("leaf-order box gather differs from primitive IDs");
      }
    }
    struct Pending { uint32_t index, depth; };
    std::vector<Pending> pending{{0,0}};
    std::vector<unsigned char> seen(nodes.size(),0);
    uint32_t maxDepth = 0;
    while (!pending.empty()) {
      const Pending current = pending.back();
      pending.pop_back();
      if (current.index >= nodes.size() || seen[current.index]++)
        throw std::runtime_error("invalid or repeated reachable BVH node");
      const auto &node = nodes[current.index];
      if (!validBounds(node.bounds))
        throw std::runtime_error("invalid BVH node bounds");
      maxDepth = std::max(maxDepth,current.depth);
      if (current.depth > stackCapacity)
        throw std::runtime_error("BVH exceeds traversal stack capacity");
      const Admin admin = node.admin;
      if (admin.count) {
        if (admin.offset+admin.count > bvh.numPrims)
          throw std::runtime_error("invalid BVH leaf range");
        for (uint64_t i=admin.offset;i<admin.offset+admin.count;++i)
          if (coverage[size_t(i)]++
              || !contains(node.bounds,reordered ? gathered[size_t(i)] : boxes[ids[size_t(i)]]))
            throw std::runtime_error("overlapping leaf ranges or insufficient leaf bounds");
      } else {
        if (admin.offset < 2 || (admin.offset&1) || admin.offset+1 >= nodes.size())
          throw std::runtime_error("invalid BVH child pair");
        for (uint32_t child=0;child<2;++child) {
          const uint32_t index = uint32_t(admin.offset+child);
          if (!contains(node.bounds,nodes[index].bounds))
            throw std::runtime_error("BVH parent does not contain child bounds");
          pending.push_back({index,current.depth+1});
        }
      }
    }
    if (std::find(coverage.begin(),coverage.end(),0) != coverage.end())
      throw std::runtime_error("BVH leaves do not cover every primitive");
    std::cout << "tree_audit=PASS max_depth=" << maxDepth
              << " stack_capacity=" << stackCapacity << '\n';
  }

  QueryResult bruteQuery(uint32_t id, const std::vector<box2f> &boxes, Grid grid)
  {
    const auto query = queryBox(id,grid);
    QueryResult result;
    for (const auto &box : boxes) {
      const int64_t dx = int64_t(std::min(box.upper.x,query.upper.x))
        -int64_t(std::max(box.lower.x,query.lower.x));
      const int64_t dy = int64_t(std::min(box.upper.y,query.upper.y))
        -int64_t(std::max(box.lower.y,query.lower.y));
      if (dx > 0 && dy > 0) {
        result.area += uint64_t(dx*dy);
        ++result.hits;
      }
    }
    return result;
  }

  QueryResult queryOracle(int64_t x0, int64_t y0, uint32_t window,
                          uint32_t tilesX, uint32_t tilesY)
  {
    QueryResult result;
    const int64_t x1 = x0+window, y1 = y0+window;
    const int64_t lowerX = std::max(x0,int64_t(-128));
    const int64_t lowerY = std::max(y0,int64_t(-128));
    const int64_t upperX = std::min(x1,int64_t(tilesX)*eda_layout::tilePitch-128);
    const int64_t upperY = std::min(y1,int64_t(tilesY)*eda_layout::tilePitch-128);
    if (upperX <= lowerX || upperY <= lowerY)
      return result;
    const uint32_t firstX = uint32_t((lowerX+128)/eda_layout::tilePitch);
    const uint32_t firstY = uint32_t((lowerY+128)/eda_layout::tilePitch);
    const uint32_t lastX = uint32_t((upperX+127)/eda_layout::tilePitch);
    const uint32_t lastY = uint32_t((upperY+127)/eda_layout::tilePitch);
    for (uint32_t y=firstY;y<=lastY;++y)
      for (uint32_t x=firstX;x<=lastX;++x)
        for (uint32_t r=0;r<eda_layout::rectanglesPerTile;++r) {
          const auto rect = eda_layout::layoutRectangle(x,y,r);
          const int64_t originX = int64_t(x)*eda_layout::tilePitch-128;
          const int64_t originY = int64_t(y)*eda_layout::tilePitch-128;
          const int64_t dx = std::min(originX+rect.x1,x1)-std::max(originX+rect.x0,x0);
          const int64_t dy = std::min(originY+rect.y1,y1)-std::max(originY+rect.y0,y0);
          if (dx > 0 && dy > 0) {
            result.area += uint64_t(dx*dy);
            ++result.hits;
          }
        }
    return result;
  }

  std::vector<float> expectedResults(const Options &options,
                                     const std::vector<box2f> &boxes, Grid grid)
  {
    const uint32_t count = grid.x*grid.y;
    std::vector<float> expected(count);
    const bool brute = uint64_t(count)*boxes.size() <= UINT64_C(8000000);
    uint64_t totalHits = 0, emptyQueries = 0;
    uint32_t maxHits = 0;
    for (uint32_t id=0;id<count;++id) {
      const auto result = queryOracle(
        grid.originX+int64_t(id%grid.x)*grid.stride,
        grid.originY+int64_t(id/grid.x)*grid.stride,
        grid.window,options.tilesX,options.tilesY);
      if (result.area > exactIntegerLimit)
        throw std::runtime_error("query area exceeds exact float integer range");
      expected[id] = float(result.area);
      if (brute) {
        const auto independent = bruteQuery(id,boxes,grid);
        if (independent.area != result.area || independent.hits != result.hits)
          throw std::runtime_error("integer fixture oracle differs from brute force");
      }
      totalHits += result.hits;
      maxHits = std::max(maxHits,result.hits);
      emptyQueries += result.hits == 0;
    }
    std::cout << "cpu_oracle=all_queries brute_force=" << (brute ? "all_queries_PASS" : "small_tests_only")
              << " empty_queries=" << emptyQueries << " mean_hits=" << double(totalHits)/count
              << " max_hits=" << maxHits << '\n';
    return expected;
  }

  uint64_t outputFingerprint(const std::vector<float> &values)
  {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (const float value : values) {
      uint32_t bits = 0;
      std::memcpy(&bits,&value,sizeof(bits));
      for (int byte=0;byte<4;++byte) {
        hash ^= (bits>>(8*byte))&255u;
        hash *= UINT64_C(1099511628211);
      }
    }
    return hash;
  }

  void validate(const std::vector<float> &expected, const float *device,
                int variant, const char *phase)
  {
    std::vector<float> actual(expected.size());
    check(cudaMemcpy(actual.data(),device,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for (size_t i=0;i<actual.size();++i)
      if (!std::isfinite(actual[i]) || actual[i] != expected[i])
        throw std::runtime_error(std::string(variantNames[variant])+" output or coverage mismatch at query "+std::to_string(i));
    std::cout << "variant=" << variantNames[variant] << " phase=" << phase
              << " queries_checked=" << actual.size() << " output_fnv64="
              << std::hex << outputFingerprint(actual) << std::dec << " correctness=PASS\n";
  }

  void initializeResults(float *results, uint32_t count)
  {
    check(cudaMemset(results,0xff,size_t(count)*sizeof(float)));
  }

  void validateRootCases()
  {
    const Grid grid{3,2,32,100};
    const auto dispatch = makeDispatch(grid);
    DeviceBuffer<uint32_t> deviceDispatch(dispatch.size()), primitiveIDs(1);
    DeviceBuffer<box2f> deviceBoxes(1);
    DeviceBuffer<bvh2f::node_t> deviceNodes(1);
    DeviceBuffer<float> output(dispatch.size());
    check(cudaMemcpy(deviceDispatch.data,dispatch.data(),dispatch.size()*sizeof(uint32_t),cudaMemcpyHostToDevice));
    const box2f box{cuBQL::vec2f{-120,-120},cuBQL::vec2f{-96,-96}};
    check(cudaMemcpy(deviceBoxes.data,&box,sizeof(box),cudaMemcpyHostToDevice));
    const uint32_t id = 0;
    check(cudaMemcpy(primitiveIDs.data,&id,sizeof(id),cudaMemcpyHostToDevice));
    bvh2f::node_t node{};
    node.bounds = box;
    node.admin.offset = 0;
    node.admin.count = 1;
    check(cudaMemcpy(deviceNodes.data,&node,sizeof(node),cudaMemcpyHostToDevice));
    bvh2f single;
    single.nodes = deviceNodes.data;
    single.numNodes = 1;
    single.primIDs = primitiveIDs.data;
    single.numPrims = 1;
    const std::vector<box2f> boxes{box};
    std::vector<float> expected(dispatch.size(),0);
    for (uint32_t query=0;query<expected.size();++query)
      expected[query] = float(bruteQuery(query,boxes,grid).area);
    for (int variant=0;variant<3;++variant) {
      initializeResults(output.data,uint32_t(dispatch.size()));
      launch(variant,single,single,deviceBoxes.data,deviceBoxes.data,grid,deviceDispatch.data,output.data);
      check(cudaDeviceSynchronize());
      validate(expected,output.data,variant,"single_root_leaf");
      initializeResults(output.data,uint32_t(dispatch.size()));
      launch(variant,bvh2f{},bvh2f{},nullptr,nullptr,grid,deviceDispatch.data,output.data);
      check(cudaDeviceSynchronize());
      validate(std::vector<float>(dispatch.size(),0),output.data,variant,"empty_tree");
    }
  }

  float measure(int variant, bvh2f reference, bvh2f optimized, const box2f *boxes,
                const box2f *reordered, Grid grid, const uint32_t *dispatch,
                float *results, Event &start, Event &stop)
  {
    initializeResults(results,grid.x*grid.y);
    check(cudaEventRecord(start.handle));
    launch(variant,reference,optimized,boxes,reordered,grid,dispatch,results);
    check(cudaEventRecord(stop.handle));
    check(cudaEventSynchronize(stop.handle));
    float milliseconds = 0;
    check(cudaEventElapsedTime(&milliseconds,start.handle,stop.handle));
    if (!std::isfinite(milliseconds) || milliseconds <= 0)
      throw std::runtime_error("invalid CUDA event interval");
    return milliseconds;
  }

  double median(std::vector<float> samples)
  {
    std::sort(samples.begin(),samples.end());
    return 0.5*(double(samples[samples.size()/2-1])+samples[samples.size()/2]);
  }

  void run(const Options &options)
  {
    const Grid grid{options.queriesX,options.queriesY,options.stride,options.window};
    const uint32_t count = grid.x*grid.y;
    int deviceID = 0, runtimeVersion = 0, driverVersion = 0;
    cudaDeviceProp device{};
    check(cudaGetDevice(&deviceID));
    check(cudaGetDeviceProperties(&device,deviceID));
    check(cudaRuntimeGetVersion(&runtimeVersion));
    check(cudaDriverGetVersion(&driverVersion));
    if (device.warpSize != warpUnified::warpSize || device.maxThreadsPerBlock < referenceBlockSize
        || uint64_t(count+1)/2 > uint64_t(device.maxGridSize[0]))
      throw std::runtime_error("device does not support the required warp or launch geometry");
    std::cout << std::fixed << std::setprecision(6) << "device=" << device.name
              << " runtime=" << runtimeVersion << " driver=" << driverVersion << '\n';
    validateRootCases();
    const auto boxes = eda_layout::makeBoxes<float,2>(options.tilesX,options.tilesY,
                                                    eda_layout::Pattern::Layout);
    std::cout << "input=shared_morton_layout tiles=" << options.tilesX << 'x' << options.tilesY
              << " boxes=" << boxes.size() << " input_bytes=" << boxes.size()*sizeof(box2f)
              << " geometry_fnv64=" << std::hex << eda_layout::fingerprint(boxes) << std::dec << '\n';
    std::cout << "queries=" << grid.x << 'x' << grid.y << " query_count=" << count
              << " origin=" << grid.originX << ',' << grid.originY << " stride=" << grid.stride
              << " window=" << grid.window << " windows_overlap=" << (grid.window > grid.stride ? "yes" : "no") << '\n';
    std::cout << "reference=radixBuilder leaf=1 max_allowed_leaf=1 block_threads=" << referenceBlockSize
              << " queries_per_thread=1 query_order=row_major box_lookup=primitive_ids\n"
              << "before=radixBuilder leaf=16 block_threads=" << blockSize
              << " queries_per_warp=1 query_order=morton box_lookup=leaf_order\n"
              << "after=radixBuilder leaf=16 block_threads=" << blockSize
              << " queries_per_warp=1 query_order=morton box_lookup=leaf_order\n";
    const auto expected = expectedResults(options,boxes,grid);
    const auto dispatch = makeDispatch(grid);
    DeviceBuffer<box2f> deviceBoxes(boxes.size()), reordered(boxes.size());
    DeviceBuffer<uint32_t> deviceDispatch(dispatch.size());
    DeviceBuffer<float> reference(count), before(count), after(count);
    float *outputs[] = {reference.data,before.data,after.data};
    check(cudaMemcpy(deviceBoxes.data,boxes.data(),boxes.size()*sizeof(box2f),cudaMemcpyHostToDevice));
    check(cudaMemcpy(deviceDispatch.data,dispatch.data(),dispatch.size()*sizeof(uint32_t),cudaMemcpyHostToDevice));
    Tree referenceTree, tree;
    auto referenceConfig = cuBQL::BuildConfig(1);
    referenceConfig.maxAllowedLeafSize = 1;
    cuBQL::cuda::radixBuilder(referenceTree.bvh,deviceBoxes.data,uint32_t(boxes.size()),
                             referenceConfig,0,referenceTree.resource);
    cuBQL::cuda::radixBuilder(tree.bvh,deviceBoxes.data,uint32_t(boxes.size()),
                             cuBQL::BuildConfig(16),0,tree.resource);
    warpUnified::gatherBoxesLeafOrderKernel<<<uint32_t((boxes.size()+blockSize-1)/blockSize),blockSize>>>(
      deviceBoxes.data,tree.bvh.primIDs,int(boxes.size()),reordered.data);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    std::cout << "variant=reference ";
    auditTree(referenceTree.bvh,boxes,nullptr,referenceStackDepth);
    std::cout << "variant=before_after ";
    auditTree(tree.bvh,boxes,reordered.data,warpUnified::stackDepth);
    for (int variant=0;variant<3;++variant) {
      for (int warmup=0;warmup<(options.benchmark ? warmupCount : 1);++warmup) {
        initializeResults(outputs[variant],count);
        launch(variant,referenceTree.bvh,tree.bvh,deviceBoxes.data,reordered.data,
               grid,deviceDispatch.data,outputs[variant]);
        check(cudaDeviceSynchronize());
      }
      validate(expected,outputs[variant],variant,"before_timing");
    }
    if (!options.benchmark) {
      std::cout << "status=PASS mode=correctness\n";
      return;
    }
    Event start, stop;
    constexpr int order[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    std::vector<float> samples[3];
    for (int round=0;round<roundCount;++round)
      for (int position=0;position<3;++position) {
        const int variant = order[round%6][position];
        const float milliseconds = measure(variant,referenceTree.bvh,tree.bvh,deviceBoxes.data,
                                           reordered.data,grid,deviceDispatch.data,outputs[variant],start,stop);
        samples[variant].push_back(milliseconds);
        std::cout << "round=" << round+1 << " position=" << position+1
                  << " variant=" << variantNames[variant] << " query_kernel_ms=" << milliseconds << '\n';
      }
    for (int variant=0;variant<3;++variant)
      validate(expected,outputs[variant],variant,"after_timing");
    const double referenceMedian = median(samples[0]);
    const double beforeMedian = median(samples[1]), afterMedian = median(samples[2]);
    std::cout << "median_reference_ms=" << referenceMedian << " median_before_ms=" << beforeMedian
              << " median_after_ms=" << afterMedian << " reference_to_andrea=" << referenceMedian/beforeMedian
              << " reference_to_lane0=" << referenceMedian/afterMedian << " andrea_to_lane0=" << beforeMedian/afterMedian
              << " reference_to_andrea_reduction_pct=" << 100*(1-beforeMedian/referenceMedian)
              << " reference_to_lane0_reduction_pct=" << 100*(1-afterMedian/referenceMedian)
              << " andrea_to_lane0_reduction_pct=" << 100*(1-afterMedian/beforeMedian) << '\n';
    std::cout << "status=PASS timing_scope=query_kernel_only balanced_triplets=" << roundCount
              << " warmups_per_variant=" << warmupCount
              << " reference=scalar_row_major_radix_leaf1_block128"
                 " before=bounds_broadcast_all_lane_overlap after=lane0_overlap_broadcast\n";
  }
}

int main(int argc, char **argv)
{
  try {
    const auto options = parseOptions(argc,argv);
    if (options.help)
      printHelp();
    else
      run(options);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "warp_unified failed: " << error.what() << '\n';
    return 1;
  }
}
