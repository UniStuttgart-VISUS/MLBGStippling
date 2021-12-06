project(playground LANGUAGES CXX)

# Add Qt.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
find_package(Qt6 COMPONENTS Core Widgets Svg OpenGLWidgets QUIET)
if(NOT Qt6_FOUND)
    find_package(Qt5 COMPONENTS Core Widgets Svg REQUIRED)
endif()

# Locate sources.
file(GLOB_RECURSE CXX_SOURCE_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
file(GLOB_RECURSE CXX_HEADER_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")

# Define executable target.
add_executable(${PROJECT_NAME} ${CXX_SOURCE_FILES} ${CXX_HEADER_FILES} resources.qrc)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt::Widgets
    Qt::Svg
	$<$<BOOL:${Qt6_FOUND}>:Qt6::OpenGLWidgets>
    libstipple)

# Source groups for Visual Studio.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PREFIX "Source Files"
    FILES ${CXX_SOURCE_FILES} ${CXX_HEADER_FILES})

# Inject deployment command to copy dependencies.
if(WIN32)
    # Locate Qt deployment tool.
    get_target_property(_qmake_path Qt::qmake IMPORTED_LOCATION)
    get_filename_component(_qt_bin_dir "${_qmake_path}" DIRECTORY)

    # Locate runtime libraries.
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
    include(InstallRequiredSystemLibraries)

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND
            "${_qt_bin_dir}/windeployqt.exe"
            $<$<CONFIG:RELEASE>:--release>
            $<$<CONFIG:RELWITHDEBINFO>:--release>
            $<$<CONFIG:DEBUG>:--debug>
            $<$<NOT:$<BOOL:${Qt6_FOUND}>>:--no-angle>
            --no-opengl-sw
            --no-compiler-runtime
            --no-system-d3d-compiler
			$<$<NOT:$<BOOL:${Qt6_FOUND}>>:--no-webkit2>
            $<$<NOT:$<VERSION_LESS:${Qt5Core_VERSION},"5.13.0">>:--no-virtualkeyboard>
            --no-quick-import
            --no-translations
            \"$<TARGET_FILE_DIR:${PROJECT_NAME}>/$<TARGET_FILE_NAME:${PROJECT_NAME}>\"
        COMMAND
            ${CMAKE_COMMAND}
            -E copy
            ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMENT "Deploying libraries for ${PROJECT_NAME}...")
endif()
