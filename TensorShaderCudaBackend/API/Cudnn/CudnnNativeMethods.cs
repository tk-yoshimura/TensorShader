using System;
using System.Runtime.InteropServices;

using TensorShaderCudaBackend.Dll;

namespace TensorShaderCudaBackend.API {
    using cudnnConvolutionBwdDataAlgo_t = Cudnn.ConvolutionBwdDataAlgo;
    using cudnnConvolutionBwdDataPreference_t = Cudnn.ConvolutionBwdDataPreference;
    using cudnnConvolutionBwdFilterAlgo_t = Cudnn.ConvolutionBwdFilterAlgo;
    using cudnnConvolutionBwdFilterPreference_t = Cudnn.ConvolutionBwdFilterPreference;
    using cudnnConvolutionFwdAlgo_t = Cudnn.ConvolutionFwdAlgo;
    using cudnnConvolutionFwdPreference_t = Cudnn.ConvolutionFwdPreference;
    using cudnnConvolutionMode_t = Cudnn.ConvolutionMode;
    using cudnnDataType_t = TensorShaderCudaBackend.Cudnn.DataType;
    using cudnnStatus_t = Cudnn.Status;
    using cudnnTensorFormat_t = TensorShaderCudaBackend.Cudnn.TensorFormat;
    using size_t = Int64;

    public static partial class Cudnn {

#pragma warning disable IDE1006
        private static class NativeMethods {
            public static class Version7 {
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

            public static class Version8 {

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate cudnnStatus_t cudnnCreate(ref IntPtr handle);
                public static NativeMethod<cudnnCreate> CudnnCreate { get; }
                    = new NativeMethod<cudnnCreate>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnCreate));

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate cudnnStatus_t cudnnDestroy(IntPtr handle);
                public static NativeMethod<cudnnDestroy> CudnnDestroy { get; }
                    = new NativeMethod<cudnnDestroy>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnDestroy));

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate cudnnStatus_t cudnnSetStream(IntPtr handle, IntPtr streamId);
                public static NativeMethod<cudnnSetStream> CudnnSetStream { get; }
                    = new NativeMethod<cudnnSetStream>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnSetStream));

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate cudnnStatus_t cudnnCreateTensorDescriptor(ref IntPtr tensorDesc);
                public static NativeMethod<cudnnCreateTensorDescriptor> CudnnCreateTensorDescriptor { get; }
                    = new NativeMethod<cudnnCreateTensorDescriptor>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnCreateTensorDescriptor));

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate cudnnStatus_t cudnnDestroyTensorDescriptor(IntPtr tensorDesc);
                public static NativeMethod<cudnnDestroyTensorDescriptor> CudnnDestroyTensorDescriptor { get; }
                    = new NativeMethod<cudnnDestroyTensorDescriptor>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnDestroyTensorDescriptor));

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate cudnnStatus_t cudnnSetTensor4dDescriptor(
                    IntPtr tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType,
                    int n, int c, int h, int w
                );
                public static NativeMethod<cudnnSetTensor4dDescriptor> CudnnSetTensor4dDescriptor { get; }
                    = new NativeMethod<cudnnSetTensor4dDescriptor>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnSetTensor4dDescriptor));

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate cudnnStatus_t cudnnCreateFilterDescriptor(ref IntPtr filterDesc);
                public static NativeMethod<cudnnCreateFilterDescriptor> CudnnCreateFilterDescriptor { get; }
                    = new NativeMethod<cudnnCreateFilterDescriptor>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnCreateFilterDescriptor));

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate cudnnStatus_t cudnnDestroyFilterDescriptor(IntPtr filterDesc);
                public static NativeMethod<cudnnDestroyFilterDescriptor> CudnnDestroyFilterDescriptor { get; }
                    = new NativeMethod<cudnnDestroyFilterDescriptor>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnDestroyFilterDescriptor));

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate cudnnStatus_t cudnnSetFilter4dDescriptor(
                    IntPtr filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format,
                    int k, int c, int h, int w
                );
                public static NativeMethod<cudnnSetFilter4dDescriptor> CudnnSetFilter4dDescriptor { get; }
                    = new NativeMethod<cudnnSetFilter4dDescriptor>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnSetFilter4dDescriptor));

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate cudnnStatus_t cudnnCreateConvolutionDescriptor(ref IntPtr convDesc);
                public static NativeMethod<cudnnCreateConvolutionDescriptor> CudnnCreateConvolutionDescriptor { get; }
                    = new NativeMethod<cudnnCreateConvolutionDescriptor>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnCreateConvolutionDescriptor));

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate cudnnStatus_t cudnnDestroyConvolutionDescriptor(IntPtr convDesc);
                public static NativeMethod<cudnnDestroyConvolutionDescriptor> CudnnDestroyConvolutionDescriptor { get; }
                    = new NativeMethod<cudnnDestroyConvolutionDescriptor>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnDestroyConvolutionDescriptor));

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
                    = new NativeMethod<cudnnSetConvolution2dDescriptor>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnSetConvolution2dDescriptor));

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
                    IntPtr handle,
                    IntPtr xDesc, IntPtr wDesc, IntPtr convDesc, IntPtr yDesc,
                    cudnnConvolutionFwdAlgo_t algo,
                    ref size_t sizeInBytes
                );
                public static NativeMethod<cudnnGetConvolutionForwardWorkspaceSize> CudnnGetConvolutionForwardWorkspaceSize { get; }
                    = new NativeMethod<cudnnGetConvolutionForwardWorkspaceSize>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnGetConvolutionForwardWorkspaceSize));

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx(
                    IntPtr handle,
                    IntPtr xDesc, IntPtr x,
                    IntPtr wDesc, IntPtr w,
                    IntPtr convDesc,
                    IntPtr yDesc, IntPtr y,
                    int requestedAlgoCount,
                    ref int returnedAlgoCount,
                    IntPtr perfResults,
                    IntPtr workSpace,
                    size_t workSpaceSizeInBytes
                );
                public static NativeMethod<cudnnFindConvolutionForwardAlgorithmEx> CudnnFindConvolutionForwardAlgorithmEx { get; }
                    = new NativeMethod<cudnnFindConvolutionForwardAlgorithmEx>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnFindConvolutionForwardAlgorithmEx));

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
                    = new NativeMethod<cudnnConvolutionForward>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnConvolutionForward));

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
                    IntPtr handle,
                    IntPtr xDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr dwDesc,
                    cudnnConvolutionBwdFilterAlgo_t algo,
                    ref size_t sizeInBytes
                );
                public static NativeMethod<cudnnGetConvolutionBackwardFilterWorkspaceSize> CudnnGetConvolutionBackwardFilterWorkspaceSize { get; }
                    = new NativeMethod<cudnnGetConvolutionBackwardFilterWorkspaceSize>(CudaDll.CudnnSubset.cnn_train, nameof(cudnnGetConvolutionBackwardFilterWorkspaceSize));

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx(
                    IntPtr handle,
                    IntPtr xDesc, IntPtr x,
                    IntPtr dyDesc, IntPtr y,
                    IntPtr convDesc,
                    IntPtr dwDesc, IntPtr dw,
                    int requestedAlgoCount,
                    ref int returnedAlgoCount,
                    IntPtr perfResults,
                    IntPtr workSpace,
                    size_t workSpaceSizeInBytes
                );
                public static NativeMethod<cudnnFindConvolutionBackwardFilterAlgorithmEx> CudnnFindConvolutionBackwardFilterAlgorithmEx { get; }
                    = new NativeMethod<cudnnFindConvolutionBackwardFilterAlgorithmEx>(CudaDll.CudnnSubset.cnn_train, nameof(cudnnFindConvolutionBackwardFilterAlgorithmEx));

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
                    = new NativeMethod<cudnnConvolutionBackwardFilter>(CudaDll.CudnnSubset.cnn_train, nameof(cudnnConvolutionBackwardFilter));

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
                    IntPtr handle,
                    IntPtr wDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr dxDesc,
                    cudnnConvolutionBwdDataAlgo_t algo,
                    ref size_t sizeInBytes
                );
                public static NativeMethod<cudnnGetConvolutionBackwardDataWorkspaceSize> CudnnGetConvolutionBackwardDataWorkspaceSize { get; }
                    = new NativeMethod<cudnnGetConvolutionBackwardDataWorkspaceSize>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnGetConvolutionBackwardDataWorkspaceSize));

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx(
                    IntPtr handle,
                    IntPtr wDesc, IntPtr w,
                    IntPtr dyDesc, IntPtr dy,
                    IntPtr convDesc,
                    IntPtr dxDesc, IntPtr dx,
                    int requestedAlgoCount,
                    ref int returnedAlgoCount,
                    IntPtr perfResults,
                    IntPtr workSpace,
                    size_t workSpaceSizeInBytes
                );
                public static NativeMethod<cudnnFindConvolutionBackwardDataAlgorithmEx> CudnnFindConvolutionBackwardDataAlgorithmEx { get; }
                    = new NativeMethod<cudnnFindConvolutionBackwardDataAlgorithmEx>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnFindConvolutionBackwardDataAlgorithmEx));

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
                    = new NativeMethod<cudnnConvolutionBackwardData>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnConvolutionBackwardData));

                [UnmanagedFunctionPointer(CallingConvention.StdCall)]
                public delegate IntPtr cudnnGetErrorString(cudnnStatus_t status);
                public static NativeMethod<cudnnGetErrorString> CudnnGetErrorString { get; }
                    = new NativeMethod<cudnnGetErrorString>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnGetErrorString));
            }
        }
#pragma warning restore IDE1006
    }
}
