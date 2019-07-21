cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(smile)

find_package(OpenCV REQUIRED)       # 查找OpenCV

INCLUDE_DIRECTORIES(/usr/local/include)
INCLUDE_DIRECTORIES(/usr/local/include/opencv4)
INCLUDE_DIRECTORIES(/usr/local/include/opencv4/opencv2)


# message(STATUS "OpenCV library status:")
# message(STATUS "    version: ${OpenCV_VERSION}")
# message(STATUS "    libraries: ${OpenCV_LIBS}")
# message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(smile image_correction.cpp)
target_link_libraries(smile ${TORCH_LIBRARIES} ${OpenCV_LIBS}) 
set_property(TARGET smile PROPERTY CXX_STANDARD 17)