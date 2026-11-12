# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if (CUBQL_HAVE_CUDA)

  add_executable(cubql_warp_unified warp_unified.cu)
  target_link_libraries(cubql_warp_unified PRIVATE cuBQL)
  target_compile_definitions(cubql_warp_unified PRIVATE CUBQL_HAVE_CUDA=1)
  set_target_properties(cubql_warp_unified PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  add_test(NAME cubql_warp_unified
    COMMAND cubql_warp_unified --tiles 8 4)
  add_test(NAME cubql_warp_unified_tails
    COMMAND cubql_warp_unified --tiles 9 7 --queries 31 15 --window 128)
  add_test(NAME cubql_warp_unified_small_windows
    COMMAND cubql_warp_unified --tiles 8 4 --queries 147 74 --stride 7 --window 7)
  add_test(NAME cubql_warp_unified_regions
    COMMAND cubql_warp_unified --tiles 64 64 --queries 7 7 --stride 1024 --window 128)
  add_test(NAME cubql_warp_unified_single_query
    COMMAND cubql_warp_unified --tiles 1 1 --queries 1 1 --window 512)
  add_test(NAME cubql_warp_unified_boundaries
    COMMAND cubql_warp_unified --tiles 3 2 --queries 7 5 --stride 37 --window 1)
  add_test(NAME cubql_warp_unified_dense
    COMMAND cubql_warp_unified --tiles 16 16 --queries 15 13 --stride 127 --window 512)

  add_executable(cubql_morton_sort_range morton_sort_range.cu)
  target_link_libraries(cubql_morton_sort_range PRIVATE cuBQL)
  target_compile_definitions(cubql_morton_sort_range PRIVATE CUBQL_HAVE_CUDA=1)
  set_target_properties(cubql_morton_sort_range PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  add_test(NAME cubql_morton_sort_range COMMAND cubql_morton_sort_range)

endif()
