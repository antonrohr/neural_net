cmake_minimum_required(VERSION 3.10)

# --- Project specific data
project(neural_net)

set(PROJECT_SOURCES src/main.cpp src/neural_net.hpp src/neural_net.cpp src/node.hpp src/node.cpp src/layer.hpp src/layer.cpp src/input_layer.hpp src/input_layer.cpp)


# --- Project specific packages
# find_package( OpenCV REQUIRED )
# set(PROJECT_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
# set(PROJECT_LIBS ${OpenCV_LIBS})
find_package(MNIST PATHS ../mnist)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)
set(PROJECT_INCLUDE_DIRS ${MNIST_INCLUDE_DIR})
set(PROJECT_DEFINITIONS MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

# --- always the same, do not change
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_FLAGS "-std=c++14 -fstrict-overflow -Werror -Wall -Wshadow -Wstrict-overflow=5 -pedantic")
set(CMAKE_CXX_FLAGS "-fstrict-overflow -Werror -Wall -Wshadow -Wstrict-overflow=5 -pedantic")

include_directories( ${PROJECT_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

target_link_libraries( ${PROJECT_NAME} ${PROJECT_LIBS} )

target_compile_definitions( ${PROJECT_NAME} PRIVATE ${PROJECT_DEFINITIONS} )