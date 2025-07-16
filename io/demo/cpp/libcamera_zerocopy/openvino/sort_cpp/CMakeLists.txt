cmake_minimum_required(VERSION 3.10)
project(YOLOv5_Object_Tracking_CPP)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find Dependencies ---
# You might need to provide the paths to OpenCV and OpenVINO when running cmake
# Example:
# cmake .. -DOpenCV_DIR=/path/to/opencv/build -DOpenVINO_DIR=/path/to/openvino/runtime/cmake

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please specify OpenCV_DIR.")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenVINO REQUIRED)
if(NOT OpenVINO_FOUND)
    message(FATAL_ERROR "OpenVINO not found. Please specify OpenVINO_DIR.")
endif()
include_directories(${OpenVINO_INCLUDE_DIRS})


# --- Define Source Files ---
set(SORT_SOURCES
    Sort.cpp
)

# --- Build Visual Version ---
add_executable(yolo_visual main_visual.cpp ${SORT_SOURCES})
target_link_libraries(yolo_visual PRIVATE ${OpenCV_LIBS} openvino::runtime)

# --- Build Console Version ---
add_executable(yolo_console main_console.cpp ${SORT_SOURCES})
target_link_libraries(yolo_console PRIVATE ${OpenCV_LIBS} openvino::runtime)


# --- Build Instructions ---
message(STATUS "Configuration successful.")
message(STATUS "To build the project, run the following commands:")
message(STATUS "  mkdir build")
message(STATUS "  cd build")
message(STATUS "  cmake .. -DOpenCV_DIR=<path_to_opencv_build> -DOpenVINO_DIR=<path_to_openvino_sdk>/runtime/cmake")
message(STATUS "  cmake --build .")
message(STATUS "Executable 'yolo_visual' and 'yolo_console' will be in the 'build' directory.")
