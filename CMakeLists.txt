cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(id)

find_package(OpenCV REQUIRED)       # 查找OpenCV

INCLUDE_DIRECTORIES(/usr/local/include)
INCLUDE_DIRECTORIES(/usr/local/include/opencv4)
INCLUDE_DIRECTORIES(/usr/local/include/opencv4/opencv2)


# message(STATUS "OpenCV library status:")
# message(STATUS "    version: ${OpenCV_VERSION}")
# message(STATUS "    libraries: ${OpenCV_LIBS}")
# message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(id example/utils/demo.cc)
target_link_libraries(id ${TORCH_LIBRARIES} ${OpenCV_LIBS}) 
set_property(TARGET id PROPERTY CXX_STANDARD 17)