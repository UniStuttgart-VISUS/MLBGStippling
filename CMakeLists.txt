cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(MLBGStippling)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY AUTOGEN_TARGETS_FOLDER "Autogen Targets")
set_property(GLOBAL PROPERTY AUTOGEN_SOURCE_GROUP "Generated Files")

if(MSVC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.20)
    # Silence external warnings (Visual Studio 2019+)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /experimental:external /external:W0 ")
    set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "/external:I ")
endif()

add_subdirectory(libstipple)
add_subdirectory(playground)
