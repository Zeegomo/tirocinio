cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(rnn)

if(UNIX)
	SET(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/pytorch-lib/unix/libtorch/")
else()
	SET(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}\\pytorch-lib\\windows\\libtorch")
endif()

find_package(Torch REQUIRED)


add_executable(py-rnn ../common/rnn.cpp ../common/error.cpp pytorch_layers.cpp main.cpp pytorch_utils.cpp ../common/csv.cpp pytorch.cpp ../common/network.cpp)
target_include_directories(py-rnn PRIVATE ../common/ )

target_link_libraries(py-rnn "${TORCH_LIBRARIES}")
target_compile_options(py-rnn PUBLIC "-g")
