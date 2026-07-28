cmake_minimum_required(VERSION 3.20)

project(SingleInstanceDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "C:/Libraries/opencv/build")
find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_DIR "C:/Libraries/onnxruntime")

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

add_executable(SingleInstanceDetection
    src/main.cpp
    src/VideoPlayer.cpp
    src/YOLODetector.cpp
    src/OSTrackTracker.cpp
    src/VehicleMatcher.cpp
    src/PersonMatcher.cpp
    src/ImageUtils.cpp
)

target_link_libraries(SingleInstanceDetection
    PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
)

add_custom_command(TARGET SingleInstanceDetection POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${ONNXRUNTIME_DIR}/lib"
        $<TARGET_FILE_DIR:SingleInstanceDetection>
)