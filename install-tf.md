Install TensorFlow on Mac from its source

# Dependencies
    - bazel (>0.3)
    - six
    - numpy
    - wheel
    - protoc (>=3.0)

```sh
$ sudo pip install six numpy wheel
$ brew uninstall bazel
$ brew install bazel

if protoc is not installed
$ brew install protobuf # the latest version

else

$ cd protobuf-3.3.0/python  # https://github.com/google/protobuf/releases # source
$ python setup.py build
$ python setup.py test
$ cd .. && ./autogen.sh && ./configure
$ make && sudo make install
$ protoc --version
libprotoc 3.3.0

$ cd python
$ sudo python setup.py install
```

# TensorFlow

```sh
$ git clone https://github.com/tensorflow/tensorflow
$ cd tensorflow
$ git checkout branch_name

$ ./configure
$ bazel build --config=opt --copt=-msse3  //tensorflow/tools/pip_package:build_pip_package
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
$ sudo pip install /tmp/tensorflow_pkg/tensorflow-xxxx.whl
```

*NOTE*: TensorFlow depends on protobuf (>=3.0), everything related to the lower-version should be removed clearly, otherwise the error `__init__() unexpected key 'syntax' would occur`

# TensorBoard

Has been installed as part of TensorFlow. But how to use it?

start it

```sh
$ tensorboard --logdir=/path/to/log-directory
```

where `logdir` is the directory storing the data wrote by `SummaryWriter`.

