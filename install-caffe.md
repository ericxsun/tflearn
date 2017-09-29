Install Caffe on Mac 

# Basic info

- OS X 10.10.5
- clang: Apple LLVM version 7.0.2 (clang-700.1.81)

# Dependencies
    - global dependencies
    - CUDA 7.0
    - OpenCV (>= 2.4 including 3.0)

# Install

install it and check:
```sh
$ clang --version

Apple LLVM version 6.1.0 (clang-602.0.53) (based on LLVM 3.6.0svn)

```

## global dependencies
```sh
$ brew install -vd snappy leveldb gflags glog szip lmdb
$ brew install --build-from-source --with-python -vd protobuf
```

boost 1.57
```sh
$ cd /usr/local/Homebrew/Library/Taps/homebrew/homebrew-core/Formula
$ mv boost.rb boost.rb.bak
$ wget https://raw.githubusercontent.com/Homebrew/homebrew/6fd6a9b6b2f56139a44dd689d30b7168ac13effb/Library/Formula/boost.rb

$ mv boost-python.rb boost-python.rb.bak
$ wget https://raw.githubusercontent.com/Homebrew/homebrew/3141234b3473717e87f3958d4916fe0ada0baba9/Library/Formula/boost-python.rb

$ wget https://downloads.sourceforge.net/project/boost/boost/1.57.0/boost_1_57_0.tar.bz2
$ shasum -a 256 boost_1_57_0.tar.bz2

$ brew edit boost.rb

sha1 "???" => sha256 "???"

bottle do
    cellar :any
    # sha1 "5eaa834239277ba3fabdf0f6664400e4e2ff29b4" => :yosemite
    # sha1 "4475c631c1107d50a4da54db5d5cbf938b890a9a" => :mavericks
    # sha1 "4ba6d875fe24548b8af3c0b6631ded562d2da40f" => :mountain_lion
end

$ brew edit boost-python.rb

sha1 "???" => sha256 "???"


$ brew install --build-from-source --fresh -vd  boost boost-python
```

## CUDA
[cuda 7.0](https://developer.nvidia.com/cuda-toolkit-70)


Set ENV
```sh
$ vi ~/.zshrc

export PATH=/Developer/NVIDIA/CUDA-7.0/bin${PATH:+:${PATH}}
export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-7.0/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}

$ source ~/.zshrc
```

## OpenCV
```sh
$ brew install --fresh -vd opencv
```


## Caffe

```sh
$ git clone https://github.com/BVLC/caffe.git
$ cd caffe
$ cp Makefile.config.example Makefile.config
$ vi Makefile.config

CPU_ONLY := 1
OPENCV_VERSION := 3
CUSTOM_CXX := g++

# uncomment BLAS_INCLUDE, BLAS_LIB
# Homebrew puts openblas in a directory that is not on the standard search path
BLAS_INCLUDE := $(shell brew --prefix openblas)/include
BLAS_LIB := $(shell brew --prefix openblas)/lib

# comment the ARCH
CUDA_ARCH := \
		-gencode arch=compute_20,code=sm_21 \
		-gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 


$ make all -j4
$ make runtest
```

## pyCaffe
```sh
$ vi Makefile.config
PYTHON_INCLUDE += $(dir $(shell python -c 'import numpy.core; print(numpy.core.__file__)'))/include
PYTHON_LIB += $(shell brew --prefix numpy)/lib

$ make clean && make all -j4 && make runtest && make pycaffe
```


