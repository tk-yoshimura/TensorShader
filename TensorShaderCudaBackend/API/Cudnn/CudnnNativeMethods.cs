using System;
using System.Runtime.InteropServices;

using TensorShaderCudaBackend.Dll;

namespace TensorShaderCudaBackend.API {
    using cudnnStatus_t = Cudnn.Status;
    using cudnnTensorFormat_t = Cudnn.TensorFormat;
    using cudnnDataType_t = Cudnn.DataType;
    using cudnnConvolutionMode_t = Cudnn.ConvolutionMode;
    using cudnnConvolutionFwdAlgo_t = Cudnn.ConvolutionFwdAlgo;
    using cudnnConvolutionFwdPreference_t = Cudnn.ConvolutionFwdPreference;
    using cudnnConvolutionBwdFilterAlgo_t = Cudnn.ConvolutionBwdFilterAlgo;
    using cudnnConvolutionBwdFilterPreference_t = Cudnn.ConvolutionBwdFilterPreference;
    using cudnnConvolutionBwdDataAlgo_t = Cudnn.ConvolutionBwdDataAlgo;
    using cudnnConvolutionBwdDataPreference_t = Cudnn.ConvolutionBwdDataPreference;

    using size_t = Int64;

    public static partial class Cudnn {

#pragma warning disable IDE1006
        private static class NativeMethods {

            static readonly NativeDll dll = CudaDll.Cudnn;

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnCreate(ref IntPtr handle);
            public static NativeMethod<cudnnCreate> CudnnCreate { get; }
                = new NativeMethod<cudnnCreate>(dll, nameof(cudnnCreate));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnDestroy(IntPtr handle);
            public static NativeMethod<cudnnDestroy> CudnnDestroy { get; }
                = new NativeMethod<cudnnDestroy>(dll, nameof(cudnnDestroy));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnSetStream(IntPtr handle, IntPtr streamId);
            public static NativeMethod<cudnnSetStream> CudnnSetStream { get; }
                = new NativeMethod<cudnnSetStream>(dll, nameof(cudnnSetStream));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnCreateTensorDescriptor(ref IntPtr tensorDesc);
            public static NativeMethod<cudnnCreateTensorDescriptor> CudnnCreateTensorDescriptor { get; }
                = new NativeMethod<cudnnCreateTensorDescriptor>(dll, nameof(cudnnCreateTensorDescriptor));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnDestroyTensorDescriptor(IntPtr tensorDesc);
            public static NativeMethod<cudnnDestroyTensorDescriptor> CudnnDestroyTensorDescriptor { get; }
                = new NativeMethod<cudnnDestroyTensorDescriptor>(dll, nameof(cudnnDestroyTensorDescriptor));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnSetTensor4dDescriptor(
                IntPtr tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType,
                int n, int c, int h, int w
            );
            public static NativeMethod<cudnnSetTensor4dDescriptor> CudnnSetTensor4dDescriptor { get; }
                = new NativeMethod<cudnnSetTensor4dDescriptor>(dll, nameof(cudnnSetTensor4dDescriptor));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnSetTensorNdDescriptorEx(
                IntPtr tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType,
                int nbDims, IntPtr dimA
            );
            public static NativeMethod<cudnnSetTensorNdDescriptorEx> CudnnSetTensorNdDescriptorEx { get; }
                = new NativeMethod<cudnnSetTensorNdDescriptorEx>(dll, nameof(cudnnSetTensorNdDescriptorEx));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnCreateFilterDescriptor(ref IntPtr filterDesc);
            public static NativeMethod<cudnnCreateFilterDescriptor> CudnnCreateFilterDescriptor { get; }
                = new NativeMethod<cudnnCreateFilterDescriptor>(dll, nameof(cudnnCreateFilterDescriptor));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnDestroyFilterDescriptor(IntPtr filterDesc);
            public static NativeMethod<cudnnDestroyFilterDescriptor> CudnnDestroyFilterDescriptor { get; }
                = new NativeMethod<cudnnDestroyFilterDescriptor>(dll, nameof(cudnnDestroyFilterDescriptor));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnSetFilter4dDescriptor(
                IntPtr filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format,
                int k, int c, int h, int w
            );
            public static NativeMethod<cudnnSetFilter4dDescriptor> CudnnSetFilter4dDescriptor { get; }
                = new NativeMethod<cudnnSetFilter4dDescriptor>(dll, nameof(cudnnSetFilter4dDescriptor));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnSetFilterNdDescriptor(
                IntPtr filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format,
                int nbDims, IntPtr filterDimA
            );
            public static NativeMethod<cudnnSetFilterNdDescriptor> CudnnSetFilterNdDescriptor { get; }
                = new NativeMethod<cudnnSetFilterNdDescriptor>(dll, nameof(cudnnSetFilterNdDescriptor));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnCreateConvolutionDescriptor(ref IntPtr convDesc);
            public static NativeMethod<cudnnCreateConvolutionDescriptor> CudnnCreateConvolutionDescriptor { get; }
                = new NativeMethod<cudnnCreateConvolutionDescriptor>(dll, nameof(cudnnCreateConvolutionDescriptor));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnDestroyConvolutionDescriptor(IntPtr convDesc);
            public static NativeMethod<cudnnDestroyConvolutionDescriptor> CudnnDestroyConvolutionDescriptor { get; }
                = new NativeMethod<cudnnDestroyConvolutionDescriptor>(dll, nameof(cudnnDestroyConvolutionDescriptor));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnSetConvolution2dDescriptor(
                IntPtr convDesc,
                int pad_h, int pad_w,          
                int stride_u, int stride_v,    
                int dilation_h, int dilation_w,
                cudnnConvolutionMode_t mode,
                cudnnDataType_t computeType
            );
            public static NativeMethod<cudnnSetConvolution2dDescriptor> CudnnSetConvolution2dDescriptor { get; }
                = new NativeMethod<cudnnSetConvolution2dDescriptor>(dll, nameof(cudnnSetConvolution2dDescriptor));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnSetConvolutionNdDescriptor(
                IntPtr convDesc,
                int arrayLength,
                IntPtr padA, IntPtr filterStrideA, IntPtr dilationA,
                cudnnConvolutionMode_t mode,
                cudnnDataType_t computeType
            ); 
            public static NativeMethod<cudnnSetConvolutionNdDescriptor> CudnnSetConvolutionNdDescriptor { get; }
                = new NativeMethod<cudnnSetConvolutionNdDescriptor>(dll, nameof(cudnnSetConvolutionNdDescriptor));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
                IntPtr handle,
                IntPtr xDesc, IntPtr wDesc, IntPtr convDesc, IntPtr yDesc,
                cudnnConvolutionFwdAlgo_t algo,
                ref size_t sizeInBytes
            );
            public static NativeMethod<cudnnGetConvolutionForwardWorkspaceSize> CudnnGetConvolutionForwardWorkspaceSize { get; }
                = new NativeMethod<cudnnGetConvolutionForwardWorkspaceSize>(dll, nameof(cudnnGetConvolutionForwardWorkspaceSize));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(
                IntPtr handle,
                IntPtr xDesc, IntPtr wDesc, IntPtr convDesc, IntPtr yDesc,
                cudnnConvolutionFwdPreference_t preference,
                size_t memoryLimitInBytes,
                ref cudnnConvolutionFwdAlgo_t algo
            );
            public static NativeMethod<cudnnGetConvolutionForwardAlgorithm> CudnnGetConvolutionForwardAlgorithm { get; }
                = new NativeMethod<cudnnGetConvolutionForwardAlgorithm>(dll, nameof(cudnnGetConvolutionForwardAlgorithm));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnConvolutionForward(
                IntPtr handle,
                IntPtr alpha,
                IntPtr xDesc, IntPtr x,
                IntPtr wDesc, IntPtr w,
                IntPtr convDesc,
                cudnnConvolutionFwdAlgo_t algo,
                IntPtr workSpace,
                size_t workSpaceSizeInBytes,
                IntPtr beta,
                IntPtr yDesc, IntPtr y
            );
            public static NativeMethod<cudnnConvolutionForward> CudnnConvolutionForward { get; }
                = new NativeMethod<cudnnConvolutionForward>(dll, nameof(cudnnConvolutionForward));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
                IntPtr handle,
                IntPtr xDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr dwDesc,
                cudnnConvolutionBwdFilterAlgo_t algo,
                ref size_t sizeInBytes
            );
            public static NativeMethod<cudnnGetConvolutionBackwardFilterWorkspaceSize> CudnnGetConvolutionBackwardFilterWorkspaceSize { get; }
                = new NativeMethod<cudnnGetConvolutionBackwardFilterWorkspaceSize>(dll, nameof(cudnnGetConvolutionBackwardFilterWorkspaceSize));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(
                IntPtr handle,
                IntPtr xDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr dwDesc,
                cudnnConvolutionBwdFilterPreference_t preference,
                size_t memoryLimitInBytes,
                ref cudnnConvolutionBwdFilterAlgo_t algo
            );
            public static NativeMethod<cudnnGetConvolutionBackwardFilterAlgorithm> CudnnGetConvolutionBackwardFilterAlgorithm { get; }
                = new NativeMethod<cudnnGetConvolutionBackwardFilterAlgorithm>(dll, nameof(cudnnGetConvolutionBackwardFilterAlgorithm));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnConvolutionBackwardFilter(
                IntPtr handle,
                IntPtr alpha,
                IntPtr xDesc, IntPtr x,
                IntPtr dyDesc, IntPtr dy,
                IntPtr convDesc,
                cudnnConvolutionBwdFilterAlgo_t algo,
                IntPtr workSpace,
                size_t workSpaceSizeInBytes,
                IntPtr beta,
                IntPtr dwDesc, IntPtr dw
            );
            public static NativeMethod<cudnnConvolutionBackwardFilter> CudnnConvolutionBackwardFilter { get; }
                = new NativeMethod<cudnnConvolutionBackwardFilter>(dll, nameof(cudnnConvolutionBackwardFilter));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
                IntPtr handle,
                IntPtr wDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr dxDesc,
                cudnnConvolutionBwdDataAlgo_t algo,
                ref size_t sizeInBytes
            );
            public static NativeMethod<cudnnGetConvolutionBackwardDataWorkspaceSize> CudnnGetConvolutionBackwardDataWorkspaceSize { get; }
                = new NativeMethod<cudnnGetConvolutionBackwardDataWorkspaceSize>(dll, nameof(cudnnGetConvolutionBackwardDataWorkspaceSize));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm(
                IntPtr handle,
                IntPtr wDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr dxDesc,
                cudnnConvolutionBwdDataPreference_t preference,
                size_t memoryLimitInBytes,
                ref cudnnConvolutionBwdDataAlgo_t algo
            );
            public static NativeMethod<cudnnGetConvolutionBackwardDataAlgorithm> CudnnGetConvolutionBackwardDataAlgorithm { get; }
                = new NativeMethod<cudnnGetConvolutionBackwardDataAlgorithm>(dll, nameof(cudnnGetConvolutionBackwardDataAlgorithm));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnConvolutionBackwardData(
                IntPtr handle,
                IntPtr alpha,
                IntPtr wDesc, IntPtr w,
                IntPtr dyDesc, IntPtr dy,
                IntPtr convDesc,
                cudnnConvolutionBwdDataAlgo_t algo,
                IntPtr workSpace,
                size_t workSpaceSizeInBytes,
                IntPtr beta,
                IntPtr dxDesc, IntPtr dx
            );
            public static NativeMethod<cudnnConvolutionBackwardData> CudnnConvolutionBackwardData { get; }
                = new NativeMethod<cudnnConvolutionBackwardData>(dll, nameof(cudnnConvolutionBackwardData));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate IntPtr cudnnGetErrorString(cudnnStatus_t status);
            public static NativeMethod<cudnnGetErrorString> CudnnGetErrorString { get; }
                = new NativeMethod<cudnnGetErrorString>(dll, nameof(cudnnGetErrorString));
        }
#pragma warning restore IDE1006
    }
}
