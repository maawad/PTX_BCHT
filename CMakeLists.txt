cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(ptx_bcht LANGUAGES CXX)

# Set c++ standard
set(CMAKE_CXX_STANDARD 17)

# Find CUDA libs and includes to link against
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

set(external_includes "${CMAKE_CURRENT_SOURCE_DIR}/external")
set(ptx_directory "${CMAKE_CURRENT_SOURCE_DIR}/ptx")
set(cuda_includes     ${CUDAToolkit_INCLUDE_DIRS})
set(includes_directory "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(ptx_cuckoo_hashtable ptx_cuckoo_hashtable.cpp)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

target_include_directories(ptx_cuckoo_hashtable
                      PRIVATE ${external_includes}
                      PRIVATE ${cuda_includes}
                      PRIVATE ${includes_directory})
target_compile_definitions(ptx_cuckoo_hashtable PRIVATE PTX_INCLUDE_DIR="${ptx_directory}")
target_link_libraries(ptx_cuckoo_hashtable CUDA::cuda_driver CUDA::cudart_static)



# visual studio setup
file(TO_NATIVE_PATH ${external_includes} external_includes_native)
file(TO_NATIVE_PATH ${includes_directory} includes_directory_native)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ptx_cuckoo_hashtable)
set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES
    $(VC_IncludePath)
    $(WindowsSDK_IncludePath)
    ${external_includes_native}
    ${includes_directory_native}
    )
