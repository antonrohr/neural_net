cmake_minimum_required(VERSION 3.10)

# --- Project specific data
project(neural_net)

set(PROJECT_SOURCES src/main.cpp src/neural_net.hpp src/neural_net.cpp src/node.hpp src/node.cpp)


# --- Project specific packages
# find_package( OpenCV REQUIRED )
# set(PROJECT_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
# set(PROJECT_LIBS ${OpenCV_LIBS})

# --- always the same, do not change
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_FLAGS "-std=c++14 -fstrict-overflow -Werror -Wall -Wshadow -Wstrict-overflow=5 -pedantic")
set(CMAKE_CXX_FLAGS "-fstrict-overflow -Werror -Wall -Wshadow -Wstrict-overflow=5 -pedantic")

include_directories( ${PROJECT_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

target_link_libraries( ${PROJECT_NAME} ${PROJECT_LIBS} )
