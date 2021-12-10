project(libstipple LANGUAGES CXX CUDA)

# Test for CUDA version.
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "11.0.0") 
    message(FATAL_ERROR "CUDA compiler version ${CMAKE_CUDA_COMPILER_VERSION} is too old.\n"
        "Please download and install CUDA version 11 or higher.")
endif()

# Locate sources.
file(GLOB_RECURSE CXX_SOURCE_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
file(GLOB_RECURSE CXX_HEADER_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
file(GLOB_RECURSE CUDA_SOURCE_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cu")
file(GLOB_RECURSE CUDA_HEADER_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cuh")
file(GLOB_RECURSE TEST_SOURCE_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test/*.cpp")
file(GLOB_RECURSE TEST_HEADER_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test/*.h")

set_source_files_properties(CUDA_SOURCE_FILES PROPERTIES LANGUAGE CUDA)

# Define library target.
add_library(${PROJECT_NAME} STATIC ${CXX_SOURCE_FILES} ${CXX_HEADER_FILES} ${CUDA_SOURCE_FILES} ${CUDA_HEADER_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "60;75" #"60;61;62;70;72;75;80;86"
    CUDA_SEPERABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
target_include_directories(${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_definitions(${PROJECT_NAME} PUBLIC _USE_MATH_DEFINES=1)
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G --source-in-ptx>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RELWITHDEBINFO>>:-lineinfo --source-in-ptx>)
target_link_libraries(${PROJECT_NAME}
    PUBLIC cuda cudadevrt cudart_static)

option(ENABLE_NVTOOLSEXT "Use NVIDIA Tools Extensions for profiling." ON)
if(ENABLE_NVTOOLSEXT)
    # Before fiddling around with detection, note that:
    # - NVTX v3 is a header-only library
    # - NVTOOLSEXT_PATH is set by an NVIDIA installer (windows)
    set(NVTOOLSEXT_PATH $ENV{NVTOOLSEXT_PATH})
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_NVTOOLSEXT=1)
    target_include_directories(${PROJECT_NAME} PUBLIC "${NVTOOLSEXT_PATH}/include")
endif()

# Define test target.
add_executable(${PROJECT_NAME}_test ${TEST_SOURCE_FILES} ${TEST_HEADER_FILES})
set_target_properties(${PROJECT_NAME}_test PROPERTIES
    CUDA_SEPERABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
target_link_libraries(${PROJECT_NAME}_test PRIVATE
    libstipple)

# Source groups for Visual Studio.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PREFIX "Source Files"
    FILES ${CXX_SOURCE_FILES} ${CXX_HEADER_FILES} ${CUDA_SOURCE_FILES} ${CUDA_HEADER_FILES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/test"
    PREFIX "Source Files"
    FILES ${TEST_SOURCE_FILES} ${TEST_HEADER_FILES})
