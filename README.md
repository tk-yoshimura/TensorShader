![TensorShader](https://github.com/tk-yoshimura/TensorShader/blob/master/logo.svg)

# TensorShader
**Deep Learning .NET library, For Regression.**

## Description
I tried to create a deep learning library of **Define and Run** / **NHWC format** from scratch.

**Supports High-Dimensional Convolution Neural Networks.** (Complex, Quaternion, Vector3D)

For regression problems: **High precision calculate** by FP32-FP32 arithmetic (1/8 error of FP32 arithmetic)

## Requirement
.NET Standard 2.0

CUDA 10.* 11.*, Compute Capability 5.0+

validated: Windows
experiment: Linux, MacOS

## Usage
[Sample](https://github.com/tk-yoshimura/TensorShader/tree/master/TensorShaderSample)

## Supported Functions
- **Connection Layers**
  - Real
    - Convolution 1D,2D,3D and Pointwise/Depthwise
    - Dense
  - Complex
    - Convolution 1D,2D,3D
    - Dense
  - Quaternion
    - Convolution 1D,2D,3D
    - Dense
  - Trivector (UnitState : Vector3D, Weight : Quaternion)
    - Convolution 1D,2D,3D
    - Dense
- **Pooling**
  - Max Average
- **Sizing**
  - ZeroPad EdgePad Trim
  - NeighborZoom LinearZoom
- **Array Manipulation**
  - Sort ArgSort Flip Reshape Concat Separate
  - ChannelToSpace SpaceToChannel 1D,2D,3D
  - ImageToColumn 1D,2D,3D
  - ExtractChannel Sum
- **Real Functions**
  - Add Sub Mul Div Abs Sign Pow Sqrt Square Cbrt Cube Rsqrt Neg Rcp
  - Sin Cos Tan Arcsin Arccos Arctan Sinh Cosh Tanh LogCosh
  - Exp Log Floor Ceil Clip Step NanAsZero Maximum Minimum
- **Complex Functions**
  - Mul Square Conjugate Squash NonlinearDecay RRelu ZRelu Normalize
- **Quaternion Functions**
  - Mul Square Conjugate Squash NonlinearDecay RRelu Normalize
- **Trivector Functions**
  - QuaternionMul Length Norm CrossProduct Squash NonlinearDecay Normalize
- **Aggregation Functions**
  - Max Min Mean Sum SquareSum SquareMean Variance
- **Logical Functions**
  - Equal NotEqual GreaterThan LessThan GreaterThanOrEqual LessThanOrEqual
  - And Or Not Xor
  - IsNan IsFinite
- **Activation Functions**
  - Relu Elu LeakyRelu SoftPlus Softmax Sigmoid
- **Loss Functions**
  - SoftmaxCrossEntropy AbsoluteError SquareError HingeLoss HuberLoss MultiBoxLoss
- **Optimizers**
  - SGD MomentumSGD NesterovAG AdaGrad AdaDelta Adam Adamax RMSprop RMSpropGraves Nadam
- **Random Generation**
  - Uniform Normal BernoulliBinary
- **Experimentals**
  - Yamatani Activation

See also... 
- [Links](https://github.com/tk-yoshimura/TensorShader/tree/master/TensorShader/Links)
- [Layers](https://github.com/tk-yoshimura/TensorShader/tree/master/TensorShader/Layers)

## Install
[Download DLL](https://github.com/tk-yoshimura/TensorShader/releases)

- To install, just import the DLL.
- This library does not change the environment at all.

## Licence
[MIT](https://github.com/tk-yoshimura/TensorShader/blob/master/LICENSE)

## Author

[tk-yoshimura](https://github.com/tk-yoshimura)

## Future Work
- Single Shot MultiBox Detector Sample
- Variational Auto Encoder Sample
- Transpose

## Troubleshooting
Can't load Cuda dll!

→Install GeForce Experience and Cuda. Check your environment variables.

→Make sure the referenced cudabackend.dll matches the installed Cuda version.

System.BadImageFormatException is thrown and DLL cannot be loaded!

→Specify x64 as the project platform.
