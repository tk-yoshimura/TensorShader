using System;
using System.Runtime.InteropServices;

using TensorShaderCudaBackend.Dll;

namespace TensorShaderCudaBackend.API {
    using cudnnConvolutionBwdDataAlgo_t = TensorShaderCudaBackend.Cudnn.ConvolutionBwdDataAlgo;
    using cudnnConvolutionBwdFilterAlgo_t = TensorShaderCudaBackend.Cudnn.ConvolutionBwdFilterAlgo;
    using cudnnConvolutionFwdAlgo_t = TensorShaderCudaBackend.Cudnn.ConvolutionFwdAlgo;
    using cudnnConvolutionMode_t = TensorShaderCudaBackend.Cudnn.ConvolutionMode;
    using cudnnDataType_t = TensorShaderCudaBackend.Cudnn.DataType;
    using cudnnStatus_t = TensorShaderCudaBackend.Cudnn.Status;
    using cudnnTensorFormat_t = TensorShaderCudaBackend.Cudnn.TensorFormat;
    using size_t = Int64;

    public static partial class Cudnn {

#pragma warning disable IDE1006
        private static class NativeMethods {

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnCreate(ref IntPtr handle);
            public static NativeMethod<cudnnCreate> CudnnCreate { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnCreate>(CudaDll.Cudnn, nameof(cudnnCreate)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnCreate>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnCreate)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnDestroy(IntPtr handle);
            public static NativeMethod<cudnnDestroy> CudnnDestroy { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnDestroy>(CudaDll.Cudnn, nameof(cudnnDestroy)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnDestroy>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnDestroy)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnSetStream(IntPtr handle, IntPtr streamId);
            public static NativeMethod<cudnnSetStream> CudnnSetStream { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnSetStream>(CudaDll.Cudnn, nameof(cudnnSetStream)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnSetStream>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnSetStream)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnCreateTensorDescriptor(ref IntPtr tensorDesc);
            public static NativeMethod<cudnnCreateTensorDescriptor> CudnnCreateTensorDescriptor { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnCreateTensorDescriptor>(CudaDll.Cudnn, nameof(cudnnCreateTensorDescriptor)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnCreateTensorDescriptor>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnCreateTensorDescriptor)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnDestroyTensorDescriptor(IntPtr tensorDesc);
            public static NativeMethod<cudnnDestroyTensorDescriptor> CudnnDestroyTensorDescriptor { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnDestroyTensorDescriptor>(CudaDll.Cudnn, nameof(cudnnDestroyTensorDescriptor)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnDestroyTensorDescriptor>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnDestroyTensorDescriptor)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnSetTensor4dDescriptor(
                IntPtr tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType,
                int n, int c, int h, int w
            );
            public static NativeMethod<cudnnSetTensor4dDescriptor> CudnnSetTensor4dDescriptor { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnSetTensor4dDescriptor>(CudaDll.Cudnn, nameof(cudnnSetTensor4dDescriptor)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnSetTensor4dDescriptor>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnSetTensor4dDescriptor)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnCreateFilterDescriptor(ref IntPtr filterDesc);
            public static NativeMethod<cudnnCreateFilterDescriptor> CudnnCreateFilterDescriptor { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnCreateFilterDescriptor>(CudaDll.Cudnn, nameof(cudnnCreateFilterDescriptor)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnCreateFilterDescriptor>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnCreateFilterDescriptor)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnDestroyFilterDescriptor(IntPtr filterDesc);
            public static NativeMethod<cudnnDestroyFilterDescriptor> CudnnDestroyFilterDescriptor { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnDestroyFilterDescriptor>(CudaDll.Cudnn, nameof(cudnnDestroyFilterDescriptor)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnDestroyFilterDescriptor>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnDestroyFilterDescriptor)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnSetFilter4dDescriptor(
                IntPtr filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format,
                int k, int c, int h, int w
            );
            public static NativeMethod<cudnnSetFilter4dDescriptor> CudnnSetFilter4dDescriptor { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnSetFilter4dDescriptor>(CudaDll.Cudnn, nameof(cudnnSetFilter4dDescriptor)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnSetFilter4dDescriptor>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnSetFilter4dDescriptor)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnCreateConvolutionDescriptor(ref IntPtr convDesc);
            public static NativeMethod<cudnnCreateConvolutionDescriptor> CudnnCreateConvolutionDescriptor { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnCreateConvolutionDescriptor>(CudaDll.Cudnn, nameof(cudnnCreateConvolutionDescriptor)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnCreateConvolutionDescriptor>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnCreateConvolutionDescriptor)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnDestroyConvolutionDescriptor(IntPtr convDesc);
            public static NativeMethod<cudnnDestroyConvolutionDescriptor> CudnnDestroyConvolutionDescriptor { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnDestroyConvolutionDescriptor>(CudaDll.Cudnn, nameof(cudnnDestroyConvolutionDescriptor)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnDestroyConvolutionDescriptor>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnDestroyConvolutionDescriptor)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnSetConvolution2dDescriptor(
                IntPtr convDesc,
                int pad_h, int pad_w,
                int stride_u, int stride_v,
                int dilation_h, int dilation_w,
                cudnnConvolutionMode_t mode,
                cudnnDataType_t computeType
            );
            public static NativeMethod<cudnnSetConvolution2dDescriptor> CudnnSetConvolution2dDescriptor { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnSetConvolution2dDescriptor>(CudaDll.Cudnn, nameof(cudnnSetConvolution2dDescriptor)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnSetConvolution2dDescriptor>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnSetConvolution2dDescriptor)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
                IntPtr handle,
                IntPtr xDesc, IntPtr wDesc, IntPtr convDesc, IntPtr yDesc,
                cudnnConvolutionFwdAlgo_t algo,
                ref size_t sizeInBytes
            );
            public static NativeMethod<cudnnGetConvolutionForwardWorkspaceSize> CudnnGetConvolutionForwardWorkspaceSize { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnGetConvolutionForwardWorkspaceSize>(CudaDll.Cudnn, nameof(cudnnGetConvolutionForwardWorkspaceSize)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnGetConvolutionForwardWorkspaceSize>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnGetConvolutionForwardWorkspaceSize)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
                IntPtr handle,
                IntPtr xDesc, IntPtr wDesc, IntPtr convDesc, IntPtr yDesc,
                int requestedAlgoCount, ref int returnedAlgoCount,
                IntPtr perfResults
            );
            public static NativeMethod<cudnnGetConvolutionForwardAlgorithm_v7> CudnnGetConvolutionForwardAlgorithm { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnGetConvolutionForwardAlgorithm_v7>(CudaDll.Cudnn, nameof(cudnnGetConvolutionForwardAlgorithm_v7)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnGetConvolutionForwardAlgorithm_v7>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnGetConvolutionForwardAlgorithm_v7)) :
                throw new NotImplementedException();

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
            public static NativeMethod<cudnnConvolutionForward> CudnnConvolutionForward { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnConvolutionForward>(CudaDll.Cudnn, nameof(cudnnConvolutionForward)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnConvolutionForward>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnConvolutionForward)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
                IntPtr handle,
                IntPtr xDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr dwDesc,
                cudnnConvolutionBwdFilterAlgo_t algo,
                ref size_t sizeInBytes
            );
            public static NativeMethod<cudnnGetConvolutionBackwardFilterWorkspaceSize> CudnnGetConvolutionBackwardFilterWorkspaceSize { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnGetConvolutionBackwardFilterWorkspaceSize>(CudaDll.Cudnn, nameof(cudnnGetConvolutionBackwardFilterWorkspaceSize)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnGetConvolutionBackwardFilterWorkspaceSize>(CudaDll.CudnnSubset.cnn_train, nameof(cudnnGetConvolutionBackwardFilterWorkspaceSize)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                IntPtr handle,
                IntPtr xDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr dwDesc,
                int requestedAlgoCount, ref int returnedAlgoCount,
                IntPtr perfResults
            );
            public static NativeMethod<cudnnGetConvolutionBackwardFilterAlgorithm_v7> CudnnGetConvolutionBackwardFilterAlgorithm { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnGetConvolutionBackwardFilterAlgorithm_v7>(CudaDll.Cudnn, nameof(cudnnGetConvolutionBackwardFilterAlgorithm_v7)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnGetConvolutionBackwardFilterAlgorithm_v7>(CudaDll.CudnnSubset.cnn_train, nameof(cudnnGetConvolutionBackwardFilterAlgorithm_v7)) :
                throw new NotImplementedException();

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
            public static NativeMethod<cudnnConvolutionBackwardFilter> CudnnConvolutionBackwardFilter { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnConvolutionBackwardFilter>(CudaDll.Cudnn, nameof(cudnnConvolutionBackwardFilter)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnConvolutionBackwardFilter>(CudaDll.CudnnSubset.cnn_train, nameof(cudnnConvolutionBackwardFilter)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
                IntPtr handle,
                IntPtr wDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr dxDesc,
                cudnnConvolutionBwdDataAlgo_t algo,
                ref size_t sizeInBytes
            );
            public static NativeMethod<cudnnGetConvolutionBackwardDataWorkspaceSize> CudnnGetConvolutionBackwardDataWorkspaceSize { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnGetConvolutionBackwardDataWorkspaceSize>(CudaDll.Cudnn, nameof(cudnnGetConvolutionBackwardDataWorkspaceSize)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnGetConvolutionBackwardDataWorkspaceSize>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnGetConvolutionBackwardDataWorkspaceSize)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
                IntPtr handle,
                IntPtr wDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr dxDesc,
                int requestedAlgoCount, ref int returnedAlgoCount,
                IntPtr perfResults
            );
            public static NativeMethod<cudnnGetConvolutionBackwardDataAlgorithm_v7> CudnnGetConvolutionBackwardDataAlgorithm { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnGetConvolutionBackwardDataAlgorithm_v7>(CudaDll.Cudnn, nameof(cudnnGetConvolutionBackwardDataAlgorithm_v7)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnGetConvolutionBackwardDataAlgorithm_v7>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnGetConvolutionBackwardDataAlgorithm_v7)) :
                throw new NotImplementedException();

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
            public static NativeMethod<cudnnConvolutionBackwardData> CudnnConvolutionBackwardData { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnConvolutionBackwardData>(CudaDll.Cudnn, nameof(cudnnConvolutionBackwardData)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnConvolutionBackwardData>(CudaDll.CudnnSubset.cnn_infer, nameof(cudnnConvolutionBackwardData)) :
                throw new NotImplementedException();

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate IntPtr cudnnGetErrorString(cudnnStatus_t status);
            public static NativeMethod<cudnnGetErrorString> CudnnGetErrorString { get; } =
                (CudaDll.CudnnVersion == 7) ? new NativeMethod<cudnnGetErrorString>(CudaDll.Cudnn, nameof(cudnnGetErrorString)) :
                (CudaDll.CudnnVersion == 8) ? new NativeMethod<cudnnGetErrorString>(CudaDll.CudnnSubset.ops_infer, nameof(cudnnGetErrorString)) :
                throw new NotImplementedException();
        }

#pragma warning restore IDE1006
    }
}
