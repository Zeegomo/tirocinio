cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(rnn)

find_package(Torch REQUIRED)
find_library(ANNT_LIBRARIES NAMES annt REQUIRED)
find_path(ANNT_INCLUDE_DIRS NAMES ANNT.hpp REQUIRED)

add_executable(rnn rnn.cpp csv.cpp pytorch.cpp network.cpp annt_net.cpp)
target_link_libraries(rnn "${TORCH_LIBRARIES}")
target_link_libraries(rnn "${ANNT_LIBRARIES}")
target_include_directories(rnn PRIVATE "${ANNT_INCLUDE_DIRS}")
target_link_options(rnn PUBLIC "-fopenmp")
target_compile_definitions(rnn PRIVATE "ANNT_USE_DOUBLE")
set_property(TARGET rnn PROPERTY CXX_STANDARD 11)
