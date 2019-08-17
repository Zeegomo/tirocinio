# tirocinio

## Requirements:
[Pytorch](https://pytorch.org/) (Libtorch C++)
[ANNT](https://github.com/cvsandbox/ANNT)


## Windows
* Clone repo
* Download and unzip [Libtorch v1.1](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.1.0.zip)
* Download and build [ANNT](https://github.com/cvsandbox/ANNT) (use MSVC v140, target x64, comment out Xparallel.h omp)
* open `CMakeLists.txt` with VS (Tested with VS 2019) and set `CMAKE_PREFIX_PATH` to hold absolute path to Libtorch and ANNT.
* build (make sure you target x64 platform, release mode)



## Unix
* Clone repo
* Download and unzip [Libtorch](https://pytorch.org/)
* Download and build [ANNT](https://github.com/cvsandbox/ANNT)
* run `cmake -DCMAKE_PREFIX_PATH="/absolute/path/to/unzipped_torch;/absolute/path/to/built_annt"`
* run `make`
* done
