project(torchvisionlib)
cmake_minimum_required(VERSION 3.21)

# accommodate differing conventions
include(GNUInstallDirs)

# We find a LibTorch installation trough the torch package.
# This is the best approach if we want to make sure we are
# targetting the same LibTorch version as used by torch.

# check if torch is installed, if it's we'll use the torch
# installation path as the libtorch path. otherwise we will
# install our own copy of libtorch.
execute_process(
  COMMAND Rscript -e "cat(as.numeric(require(torch, quietly = TRUE)))"
  OUTPUT_VARIABLE TORCH_INSTALLED
)

message(STATUS "TORCH_INSTALLED: ${TORCH_INSTALLED}")

if ("${TORCH_INSTALLED}" STREQUAL "1")
  execute_process(
      COMMAND Rscript -e "cat(torch::torch_install_path())"
      OUTPUT_VARIABLE TORCH_HOME
  )
else()
  SET(TORCH_INSTALL_SCRIPT "https://raw.githubusercontent.com/mlverse/torch/main/R/install.R")
  SET(TORCH_DESCRIPTION "https://raw.githubusercontent.com/mlverse/torch/main/DESCRIPTION")
  execute_process (
    COMMAND Rscript -e "cat(desc::desc(text=readLines('${TORCH_DESCRIPTION}'))$get('Version'))"
    OUTPUT_VARIABLE TORCH_R_VERSION
  )
  message(STATUS "TORCH_R_VERSION: ${TORCH_R_VERSION}")
  SET(TORCH_HOME "${CMAKE_CURRENT_BINARY_DIR}/torch")
  execute_process(
      COMMAND Rscript -e "Sys.setenv(TORCH_HOME='${TORCH_HOME}', TORCH_R_VERSION='${TORCH_R_VERSION}', TORCH_COMMIT_SHA='none');source('${TORCH_INSTALL_SCRIPT}'); install_torch();"
  )
endif()

message(STATUS "TORCH_HOME=${TORCH_HOME}")

if (DEFINED ENV{CUDA} AND NOT '$ENV{CUDA}' STREQUAL '')
  if (WIN32)
    find_package(CUDAToolkit)
  endif()
  enable_language(CUDA)
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${TORCH_HOME}")
set(INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib/)

include(ExternalProject)
if(WIN32)
  SET(COMMON_CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR})
  ExternalProject_Add(zlib
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG v1.2.12
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/zlib"
    CMAKE_ARGS ${COMMON_CMAKE_ARGS} -DCMAKE_ASM_COMPILER=MSVC -DCMAKE_ASM_NASM_COMPILER=MSVC
  )
  ExternalProject_Add(libpng
    GIT_REPOSITORY https://github.com/glennrp/libpng
    GIT_TAG v1.6.37
    DEPENDS zlib
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/libpng"
    CMAKE_ARGS ${COMMON_CMAKE_ARGS} -DCMAKE_IGNORE_PATH=C:/rtools40/ucrt64/include -DCMAKE_IGNORE_PATH=C:/rtools43/x86_64-w64-mingw32.static.posix/include -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH};${CURRENT_BINARY_DIR}/lib
  )
  ExternalProject_Add(libjpeg
    GIT_REPOSITORY https://github.com/libjpeg-turbo/libjpeg-turbo
    GIT_TAG 2.1.2
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/libjpeg"
    CMAKE_ARGS ${COMMON_CMAKE_ARGS} -DWITH_JPEG8=0 -DWITH_JPEG7=0 -DENABLE_SHARED=0 -DWITH_TURBOJPEG=0
  )
endif()


# Now that the prefix path is set we can tell cmake to go
# and find Torch.
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(TorchVision_DESTDIR ${CMAKE_CURRENT_BINARY_DIR}/lib/)

set(TORCHVISION_CMAKE_ARGS -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} -DCMAKE_INSTALL_PREFIX=${TorchVision_DESTDIR})
if(WIN32)
  set(TORCHVISION_CMAKE_ARGS ${TORCHVISION_CMAKE_ARGS} -DZLIB_LIBRARY=${ZLIB_LIBRARY} -DPNG_LIBRARY=${PNG_LIBRARY} -DPNG_INCLUDE_DIR=${PNG_INCLUDE_DIR} -DJPEG_LIBRARY=${JPEG_LIBRARY})
elseif(APPLE)
  set(TORCHVISION_CMAKE_ARGS ${TORCHVISION_CMAKE_ARGS} -DCMAKE_FIND_FRAMEWORK=LAST)
endif()

if (WIN32)
  set(TORCHVISION_DEPENDS libpng libjpeg)
  set(PATCH_CMD git apply ${CMAKE_SOURCE_DIR}/patch/jpeg_types.diff || true)
else()
  set(TORCHVISION_DEPENDS)
endif()

ExternalProject_Add(TorchVision-project
  GIT_REPOSITORY https://github.com/pytorch/vision
  GIT_TAG v0.14.1
  DEPENDS ${TORCHVISION_DEPENDS}
  CMAKE_ARGS ${TORCHVISION_CMAKE_ARGS}
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/libtorchvision"
  PATCH_COMMAND ${PATCH_CMD}
)

SET(TORCHVISION_INCLUDE_DIR ${TorchVision_DESTDIR}/include)
SET(TORCHVISION_LIB_DIR ${TorchVision_DESTDIR}/${CMAKE_INSTALL_LIBDIR})

if(APPLE)
  SET(TORCHVISION_LIB_LOCATION ${TORCHVISION_LIB_DIR}/libtorchvision.dylib)
elseif(WIN32)
  SET(TORCHVISION_LIB_LOCATION ${TorchVision_DESTDIR}/bin/torchvision.dll)
else()
  SET(TORCHVISION_LIB_LOCATION ${TORCHVISION_LIB_DIR}/libtorchvision.so)
endif()


add_library(TorchVision SHARED IMPORTED)
add_library(TorchVision::library ALIAS TorchVision)

set_target_properties(TorchVision PROPERTIES IMPORTED_LOCATION ${TORCHVISION_LIB_LOCATION})
if (WIN32)
  set_target_properties(TorchVision PROPERTIES IMPORTED_IMPLIB ${TorchVision_DESTDIR}/lib/torchvision.lib)
endif()

set(TORCHVISION_SRC src/torchvisionlib.cpp src/ops.cpp src/exports.cpp src/torchvisionlib_types.cpp)

add_library(torchvisionlib SHARED ${TORCHVISION_SRC})
add_library(torchvisionlib::library ALIAS torchvisionlib)

target_include_directories(torchvisionlib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${TORCH_HOME}/include
    ${TORCHVISION_INCLUDE_DIR}
)

set(TORCHVISIONLIB_HEADERS
    include/torchvisionlib/torchvisionlib.h
    include/torchvisionlib/exports.h
    include/torchvisionlib/torchvisionlib_types.h
)
set_property(TARGET torchvisionlib
 PROPERTY PUBLIC_HEADER ${TORCHVISIONLIB_HEADERS}
)

add_dependencies(torchvisionlib TorchVision-project)

add_custom_target(torchvisionlib_export
  COMMAND Rscript -e "torchexport::export()"
  COMMENT "Regenerating export code."
  VERBATIM
)
add_dependencies(torchvisionlib torchvisionlib_export)

set_property(TARGET torchvisionlib PROPERTY CXX_STANDARD 17)

target_link_libraries(torchvisionlib "${TORCH_LIBRARIES}")
target_link_libraries(torchvisionlib TorchVision)

# Syncronize the headers and the def file with the Rcpp
# interface.
add_custom_command(TARGET torchvisionlib POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/src/torchvisionlib.def ${PROJECT_SOURCE_DIR}/../inst/def/torchvisionlib.def
    COMMENT "Copied def file to inst folder."
)

# Set CPack related code to automatically generate installation bundles.
# The bundle name will have the same version as defined in the R DESCRIPTION
# file.
# The cpack configuration is used by the CI/CD workflows to create the pre-built
# binaries bundles and upload them to the GitHub Releases page.
set(CPACK_GENERATOR ZIP)
execute_process (
    COMMAND Rscript -e "cat(desc::description$new(file = '../../DESCRIPTION')$get('Version'))"
    OUTPUT_VARIABLE CPACK_PACKAGE_VERSION
)

message(STATUS "CUDA_VERSION=${CUDA_VERSION_STRING}")
if(DEFINED CUDA_VERSION_STRING)
  set(CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}+cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR})
else()
  set(CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}+cpu)
endif()

if(APPLE)
  set(CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}+${CMAKE_SYSTEM_PROCESSOR})
endif()

include(CPack)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/../inst)
install(TARGETS torchvisionlib LIBRARY PUBLIC_HEADER DESTINATION include/torchvisionlib)
install(DIRECTORY ${TorchVision_DESTDIR}/ DESTINATION .)
