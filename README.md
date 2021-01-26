# AdaTune: Adaptive Tensor Program Compilation Made Efficient

This repository is the official implementation of AdaTune: Adaptive Tensor Program Compilation Made Efficient. 

## Requirements

Install TVM first. You can find TVM installation instructions [here](https://tvm.apache.org/docs/install/from_source.html).
>Prepare llvm:
```
wget https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
tar xvJf clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz <path-to-llvm>
```

>Clone the TVM project from github:
```
git clone --recursive https://github.com/apache/incubator-tvm tvm
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
mkdir build
cp cmake/config.cmake build
```
>Edit build/config.cmake:
```
set(USE_LLVM <path-to-llvm>/bin/llvm-config)
set(USE_CUDA ON) (you can ignore this if you want to test cpu only)
```
>Building:
```
cd build
cmake ..
make -j6
```
>Add TVM into PYTHONPATH, edit your ~/.bashrc:
```
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:${PYTHONPATH}
```
>Install other required packages:
```
pip install -r requirements.txt
```
>Add AdaTune files.
```
cp tuner/* <path-to-tvm>/python/tvm/autotvm/tuner/
cp measure/measure_methods.py <path-to-tvm>/python/tvm/autotvm/measure/
```

## Optimizing Models and Evaluation

To obtain the end-to-end experiments results in the paper, run the following command:

```
python tune.py 
    --model_name <model_name>   # for example: 'resnet-18','squeezenet_v1.1','vgg-16'
    --use_gpu <use_gpu>     # bool, True/False
    --tuner <tuner>     # for example: 'ada', 'xgb'
    --ops <ops>     # for example: 'conv2d', 'dense'
```

> If the use_gpu flag is set to True, TVM should have been compiled with CUDA.
> The tune.py file will tune all the dense and conv ops in the models and then evaluate the inference latency on the optimized models. These models are constructed as TVM relay module. Please refer to the [TVM tutorial](https://tvm.apache.org/docs/tutorials/index.html) to tune more models in different formats.

### Testing environment
All the results from the paper are collected on the following hardware.
+ CPU: Intel Xeon x86 CPU E5-2690 v3
+ GPU: Nvidia Tesla P100

## Results

Our method achieves the following performance (optimization time) on the Resnet-18, VGG-16, Squeezenet_V1.1 models compared with the AutoTVM (XGBTuner):

#### Compilation time comparison
| Model name      | AutoTVM(GPU) | AdaTune(GPU) | Speedup | AutoTVM(CPU) | AdaTune(CPU) | Speedup |
| --------------- | ------------ | ------------ | ------- | ------------ | ------------ | ------- |
| Resnet-18       | 22.6h        | 9.6h         | 2.4X    | 2.0h         | 1.0h         | 2.0X    |
| Resnet-50	  | 20.0h	 | 14.1h	| 1.4X	  | 3.6h 	 | 1.7h 	| 2.1X	  |
| VGG-16          | 21.9h        | 16.7h        | 1.3X    | 18.9h        | 6.5h         | 2.9X    |
| Squeezenet_V1.1 | 7.6h         | 5.8h         | 1.3X    | 1.2h         | 0.7h         | 1.7X    |
| Encoder	  | 3.8h	 | 2.8h		| 1.4X	  | 8.4h	 | 3.8h		| 2.2X	  |

#### Inference time comparison
| Model name	  | TVM(GPU) | AutoTVM(GPU) | AdaTune(GPU) | TVM(CPU) | AutoTVM(CPU) | AdaTune(CPU) |
| --------------- | -------  | ------------ | ------------ | -------- | ------------ | ------------ |
| Resnet-18	  | 1.53ms   | 1.38ms       | 1.38ms	   | 79.24ms  | 52.64ms      | 52.64ms      |
| Resnet-50	  | 4.82ms   | 4.37ms       | 4.37ms	   | 217.12ms | 115.76ms     | 115.68ms	    |
| VGG-16          | 3.95ms   | 3.86ms       | 3.86ms	   | 884.94ms | 442.01ms     | 438.68ms     |
| Squeezenet_V1.1 | 2.93ms   | 0.65ms       | 0.63ms	   | 14.41 ms | 11.36ms	     | 11.25ms      |
| Encoder	  | 78.15ms  | 52.25ms      | 47.46ms      | 2897.27ms| 1620.88ms    | 1607.67ms    |

## Contributing
Under Apache License 2.0