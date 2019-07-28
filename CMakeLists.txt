cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(ocr)

find_package(OpenCV REQUIRED)       # 查找OpenCV

INCLUDE_DIRECTORIES(/usr/local/include)
INCLUDE_DIRECTORIES(/usr/local/include/opencv4)
INCLUDE_DIRECTORIES(/usr/local/include/opencv4/opencv2)

# message(STATUS "OpenCV library status:")
# message(STATUS "    version: ${OpenCV_VERSION}")
# message(STATUS "    libraries: ${OpenCV_LIBS}")
# message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(ocr source/main.cc source/utils/correct/rotating.cc include/correct/fourier.hpp include/correct/rotating.hpp)

target_link_libraries(ocr ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET ocr PROPERTY CXX_STANDARD 17)
set_property(TARGET ocr PROPERTY C_STANDARD 11)
