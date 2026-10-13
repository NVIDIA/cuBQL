# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)

set(CUBQL_VERSION_MAJOR 1)
set(CUBQL_VERSION_MINOR 3)
set(CUBQL_VERSION_PATCH 1)
set(CUBQL_VERSION ${CUBQL_VERSION_MAJOR}.${CUBQL_VERSION_MINOR}.${CUBQL_VERSION_PATCH})

cmake_policy(SET CMP0048 NEW)
set(CMAKE_BUILD_TYPE_INIT "Release")
project(cuBQL VERSION ${CUBQL_VERSION} LANGUAGES C CXX)

# Build the GPU code with HIP for AMD GPUs instead of CUDA. When ON, the .cu
# sources are compiled as HIP (LANGUAGE HIP) and CUBQL_HAVE_HIP is set; the
# per-type instantiation targets and GPU samples are built the same way as for
# CUDA, gated on (CUBQL_HAVE_CUDA OR CUBQL_HAVE_HIP).
option(CUBQL_USE_HIP "Build cuBQL GPU code with HIP for AMD GPUs" OFF)

# CUBQL_HAVE_CUDA / CUBQL_HAVE_HIP record which (if any) GPU toolchain is in
# use; the GPU instantiation targets and samples below are gated on either.
set(CUBQL_HAVE_CUDA OFF)
set(CUBQL_HAVE_HIP OFF)
if (CUBQL_OMP)
  set(CUBQL_DISABLE_CUDA ON)
endif()
if (CUBQL_USE_HIP)
  message("#cuBQL: building GPU code with HIP for AMD GPUs")
  enable_language(HIP)
  if (NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
    set(CMAKE_HIP_ARCHITECTURES "gfx90a")
  endif()
  set(CUBQL_HAVE_HIP ON)
elseif (CUBQL_DISABLE_CUDA)
  message("#cuBQL: CUDA _DISABLED_ by user request")
  set(CUBQL_HAVE_CUDA OFF)
else()
if (NOT CMAKE_CUDA_COMPILER)
  include(CheckLanguage)
  check_language(CUDA)
endif()

if (CMAKE_CUDA_COMPILER)
  message("#cuBQL: CUDA _FOUND_! building both cuda and host libs")
  enable_language(CUDA)
  set(CUBQL_HAVE_CUDA ON)
else()
  message(AUTHOR_WARNING
    " ===========================================================\n"
    " #cuBQL: could not find CUDA - going to build only host libs\n"
    " ===========================================================\n"
    )
  set(CUBQL_HAVE_CUDA OFF)
endif()
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 NEW)
endif()

if (NOT (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}))
  set(CUBQL_IS_SUBPROJECT ON)
else()
  set(CUBQL_IS_SUBPROJECT OFF)
endif()

# ------------------------------------------------------------------
# check if we're a subproject, and if so, only include the library
# itself
# ------------------------------------------------------------------
if (CUBQL_IS_SUBPROJECT)
  # ------------------------------------------------------------------
  # we're included from a parent project; it's this parent project's
  # job to project-specific configs like configure output paths and
  # build type, and to set CUQBL_CUDA_ARCHITECTURES to whatever
  # arch(s) the project wants us to build for.
  #
  # Check if CUBQL_CUDA_ARCHITECTURES is set, and throw an error if
  # not
  # ------------------------------------------------------------------
  if (CUBQL_HAVE_CUDA AND (NOT CMAKE_CUDA_ARCHITECTURES)
      OR
      ((${CMAKE_VERSION} VERSION_LESS 3.24) AND ("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "52"))
      )
    message(FATAL_ERROR
      " =====================================================================\n"
      " #cuBQL: you've included cuBQL as a subproject (as it should!),\n"
      " but CMAKE_CUDA_ARCHITECTURES was not set. As it is\n"
      " almost certainly going to cause some problems in your project\n"
      " if we were to just build cuBQL for archs different than your own\n"
      " project's we will cowardly refuse do so. Before including cuBQL\n"
      " you should explicitly set the cmake variable\n"
      " CMAKE_CUDA_ARCHITECTURES to whatever the main project will want\n"
      " to use, too.\n"
      
      " \n"
      
      " If in doubt as to what to use, at least for cmake >= 3.24 you can also use\n"
      "   set(CMAKE_CUDA_ARCHITECTURES \"all-major\")\n"
      "   add_subdirectory(<path-to-cubql) ...)\n"
      " or\n"
      "   set(CMAKE_CUDA_ARCHITECTURES \"native\")\n"
      "   add_subdirectory(<path-to-cubql) ...)\n"
      
      "====================================================================="
      )
  endif()
  set(CUBQL_VERSION ${CUBQL_VERSION} PARENT_SCOPE)
else()
  if (CUBQL_USE_HIP)
    # HIP build: the arch lives in CMAKE_HIP_ARCHITECTURES (defaulted above);
    # do not derive or require a CUDA arch.
  elseif (CMAKE_CUDA_ARCHITECTURES)
    # set on the cmdline, for a local build
    set(CUBQL_CUDA_ARCHITECTURES_INIT "${CMAKE_CUDA_ARCHITECTURES}")
  elseif (${CMAKE_VERSION} VERSION_LESS 3.24)
    message(AUTHOR_WARNING
      "====================================================================="
      
      "\n"

      "cuBQL: you're building with cmake version < 3.24, which doesn't
      yet support the 'native' flag for CMAKE_CUDA_ARCHITECTURES.
      We're defaulting to '70;80'; if that doesn't capture the kind of
      GPU you are using, please either specify whatever arch(s) you
      want to build for, or use a newer cmake for which we'll default
      to 'native'"
      
      "\n"

      "====================================================================="
      )
    set(CUBQL_CUDA_ARCHITECTURES_INIT "70;80")
  else()
    set(CUBQL_CUDA_ARCHITECTURES_INIT "native")
  endif()
  if (NOT CUBQL_USE_HIP)
    set(CUBQL_CUDA_ARCHITECTURES "${CUBQL_CUDA_ARCHITECTURES_INIT}" CACHE STRING
      "Which CUDA architecture to build for")
    set(CMAKE_CUDA_ARCHITECTURES "${CUBQL_CUDA_ARCHITECTURES}")
  endif()

  if(CMAKE_CONFIGURATION_TYPES) # multiconfig generator?
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE) 
  else()
    if(NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
    endif()
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
    # set the valid options for cmake-gui drop-down list
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release")
  endif()
  
  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

message("#cuBQL: compiling with CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
add_subdirectory(cuBQL)

if (NOT CUBQL_IS_SUBPROJECT)
  add_subdirectory(samples)
  option(CUBQL_BUILD_TESTS "Build cuBQL correctness tests" OFF)
  if (CUBQL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
  endif()
endif()

#add_subdirectory(testing)
