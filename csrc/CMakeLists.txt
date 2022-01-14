project(vision)
cmake_minimum_required(VERSION 3.16)

# We find a LibTorch installation trough the torch package.
# This is the best approach if we want to make sure we are
# targetting the same LibTorch version as used by torch.
execute_process (
    COMMAND Rscript -e "cat(torch::torch_install_path())"
    OUTPUT_VARIABLE TORCH_HOME
)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${TORCH_HOME}")
set(INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib/)

include(ExternalProject)
if(WIN32)
  SET(COMMON_CMAKE_ARGS -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH};${CURRENT_BINARY_DIR}/lib -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR})
  ExternalProject_Add(zlib
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG v1.2.11
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/zlib"
    CMAKE_ARGS ${COMMON_CMAKE_ARGS} -DHAVE_UNISTD_H=off
  )
  ExternalProject_Add(libpng
    GIT_REPOSITORY https://github.com/glennrp/libpng
    GIT_TAG v1.6.37
    DEPENDS zlib
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/libpng"
    CMAKE_ARGS ${COMMON_CMAKE_ARGS}

  )
  ExternalProject_Add(libjpeg
    GIT_REPOSITORY https://github.com/libjpeg-turbo/libjpeg-turbo
    GIT_TAG 2.1.2
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/libjpeg"
    CMAKE_ARGS ${COMMON_CMAKE_ARGS}
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
endif()

if (WIN32)
  set(TORCHVISION_DEPENDS libpng libjpeg)
else()
  set(TORCHVISION_DEPENDS)
endif()

ExternalProject_Add(TorchVision-project
  GIT_REPOSITORY https://github.com/dfalbel/vision
  GIT_TAG feature/no-python-cpp
  DEPENDS ${TORCHVISION_DEPENDS}
  CMAKE_ARGS ${TORCHVISION_CMAKE_ARGS}
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/libtorchvision"
)

SET(TORCHVISION_INCLUDE_DIR ${TorchVision_DESTDIR}/include)
SET(TORCHVISION_LIB_DIR ${TorchVision_DESTDIR}/lib)

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

set(TORCHVISION_SRC src/vision.cpp src/ops.cpp)

add_library(vision SHARED ${TORCHVISION_SRC})
add_library(vision::library ALIAS vision)

target_include_directories(vision PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${TORCHVISION_INCLUDE_DIR}
)

add_dependencies(vision TorchVision-project)

set_property(TARGET vision PROPERTY CXX_STANDARD 17)

target_link_libraries(vision "${TORCH_LIBRARIES}")
target_link_libraries(vision TorchVision)

# Syncronize the headers and the def file with the Rcpp
# interface.
add_custom_command(TARGET vision POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/src/vision.def ${PROJECT_SOURCE_DIR}/../inst/def/vision.def
    COMMENT "Copied def file to inst folder."
)

add_custom_command(TARGET vision POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/include/vision/vision.h ${PROJECT_SOURCE_DIR}/../src/vision/vision.h
    COMMENT "Copied vision.h file to src folder."
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

if(DEFINED ${CUDA_VERSION_STRING})
  set(CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}+cu${CUDA_VERSION_STRING})
else()
  set(CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}+cpu)
endif()


install(TARGETS vision DESTINATION lib)
include(CPack)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/../inst)
install(TARGETS vision)
install(DIRECTORY ${TorchVision_DESTDIR}/ DESTINATION .)
