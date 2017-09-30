Install Caffe on Unbuntu 

## basic info

OS: 17.04
Python: 2.7


## dependencies


```sh
$ sudo apt install libboost-all-dev libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler ibopenblas-dev h5py
```

```sh
$ cd /usr/lib/x86_64-linux-gnu
$ sudo ln -s libhdf5_serial.so.100.0.1 libhdf5.so         # /usr/bin/ld: cannot find -lhdf5
$ sudo ln -s libhdf5_serial_hl.so.100.0.0 libhdf5_hl.so   # /usr/bin/ld: cannot find -lhdf5_hl
```


## Caffe

```sh
$ git clone https://github.com/BVLC/caffe
$ cd caffe
$ cp Makefile.config.example Makefile.config

$ sudo pip install scikit-image protobuf
$ cd python
$ for req in $(cat requirements.txt); do sudo pip install $req; done

$ cd ..
$ vim Makefile.config

BLAS := open  # do not leave space after open

# CPU-only switch (uncomment to build without GPU support).
CPU_ONLY := 1

# find / -t file -name 'hdf5.h'
Add /usr/local/include and /usr/include/hdf5/serial/ after INCLUDE_DIRS := $(PYTHON_INCLUDE).

$ make all
$ make runtest

```

## PyCaffe
```sh
# locate numpy include path, may cause fatal error: numpy/arrayobject.h: No such file or directory
$ python
import numpy
numpy.get_include()
'include_path'

$ vi Makefile.config

PYTHON_INCLUDE := /usr/include/python2.7 \
                /usr/lib/python2.7/dist-packages/numpy/core/include

=>

PYTHON_INCLUDE := /usr/include/python2.7 \
                include_path

$ make pycaffe

$ vi ~/.bashrc
export CAFFE_PATH=xxxxx
export PYTHONPATH=$CAFFE_PATH/python:$PYTHONPATH

$ source ~/.bashrc

# valid
$ python -c 'import caffe'
```

## example
```sh
$ cd $CAFFE_PATH
$ ./data/mnist/get_mnist.sh
$ ./examples/mnist/create_mnist.sh
$ ./examples/mnist/train_lenet.sh

I0930 16:44:30.996455 17607 caffe.cpp:218] Using GPUs 0
F0930 16:44:30.997221 17607 common.cpp:66] Cannot use GPU in CPU-only Caffe: check mode.
*** Check failure stack trace: ***
    @     0x7fe4a8fdf5ad  google::LogMessage::Fail()
    @     0x7fe4a8fe1413  google::LogMessage::SendToLog()
    @     0x7fe4a8fdf13b  google::LogMessage::Flush()
    @     0x7fe4a8fe1dfe  google::LogMessageFatal::~LogMessageFatal()
    @     0x7fe4a9377ea0  caffe::Caffe::SetDevice()
    @     0x55af7045404e  train()
    @     0x55af7044f1bb  main
    @     0x7fe4a7e103f1  __libc_start_main
    @     0x55af7044fa5a  _start
    @              (nil)  (unknown)
Aborted

replace the GPU mode with CPU
$ vi ./examples/mnist/lenet_solver.prototxt

solver_mode: GPU

=>

solver_mode: CPU

```
