# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if (CUBQL_HAVE_CUDA)
  add_executable(cubql_morton_sort_range morton_sort_range.cu)
  target_link_libraries(cubql_morton_sort_range PRIVATE cuBQL)
  target_compile_definitions(cubql_morton_sort_range PRIVATE CUBQL_HAVE_CUDA=1)
  set_target_properties(cubql_morton_sort_range PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  add_test(NAME cubql_morton_sort_range COMMAND cubql_morton_sort_range)
endif()
