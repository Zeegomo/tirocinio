cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(rnn)

if(WIN32)
	SET(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}\\annt-lib\\windows\ ")
	SET(CMAKE_BUILD_TYPE "Release")
else()
	SET(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/annt-lib/unix/")
endif()

find_library(ANNT_LIBRARIES NAMES annt REQUIRED)
find_path(ANNT_INCLUDE_DIRS NAMES ANNT.hpp REQUIRED)

add_executable(annt-rnn ../common/rnn.cpp ../common/error.cpp annt_net.cpp ../common/csv.cpp main.cpp ../common/network.cpp)
target_include_directories(annt-rnn PRIVATE ../common/)
	
target_link_libraries(annt-rnn "${ANNT_LIBRARIES}")
target_include_directories(annt-rnn PRIVATE "${ANNT_INCLUDE_DIRS}")

if(UNIX)
	target_link_options(annt-rnn PUBLIC "-fopenmp")
endif()
