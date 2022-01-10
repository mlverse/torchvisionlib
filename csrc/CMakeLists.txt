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

# Now that the prefix path is set we can tell cmake to go
# and find Torch.
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include(ExternalProject)
set(TorchVision_DESTDIR ${CMAKE_CURRENT_BINARY_DIR}/lib/torchvision_install)

ExternalProject_Add(TorchVision
  GIT_REPOSITORY https://github.com/pytorch/vision
  GIT_TAG v0.10.0
  CMAKE_ARGS -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} -DCMAKE_INSTALL_PREFIX=${TorchVision_DESTDIR}
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
)
find_package(TorchVision REQUIRED)

set(TORCHVISION_SRC src/vision.cpp)

add_library(vision SHARED ${TORCHVISION_SRC})
add_library(vision::library ALIAS vision)

target_include_directories(vision PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

add_dependencies(vision TorchVision)

set_property(TARGET vision PROPERTY CXX_STANDARD 17)

target_link_libraries(vision "${TORCH_LIBRARIES}")
target_link_libraries(vision TorchVision::TorchVision)

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
install(TARGETS vision DESTINATION lib)
install(DIRECTORY ${TorchVision_DESTDIR}/ DESTINATION .)
