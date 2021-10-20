using System;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.API {
    public static partial class Cudnn {

        /// <summary>畳み込み</summary>
        public static class Convolution {

            /// <summary>順伝搬</summary>
            public static class Forward {

                /// <summary>アルゴリズム</summary>
                internal static ConvolutionFwdAlgo Algorithm(
                    IntPtr handle,
                    IntPtr xDesc, IntPtr wDesc, IntPtr convDesc, IntPtr yDesc,
                    ConvolutionFwdPreference preference, Int64 memoryLimitInBytes) {

                    ConvolutionFwdAlgo algo = ConvolutionFwdAlgo.ImplicitGemm;
                    
                    Status status = NativeMethods.CudnnGetConvolutionForwardAlgorithm.AsDelegate().Invoke(
                        handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, ref algo
                    );
                    if (status != Status.Success) {
                        throw new CudaException(status);
                    }

                    return algo;
                }

                /// <summary>ワークスペースサイズ算出</summary>
                internal static Int64 WorkspaceSize(
                    IntPtr handle,
                    IntPtr xDesc, IntPtr wDesc, IntPtr convDesc, IntPtr yDesc,
                    ConvolutionFwdAlgo algo) {

                    Int64 size = 0;

                    Status status = NativeMethods.CudnnGetConvolutionForwardWorkspaceSize.AsDelegate().Invoke(
                        handle, xDesc, wDesc, convDesc, yDesc, algo, ref size
                    );
                    if (status != Status.Success) {
                        throw new CudaException(status);
                    }

                    return size;
                }

                /// <summary>実行</summary>
                internal static void Execute(
                    IntPtr handle,
                    IntPtr alpha,
                    IntPtr xDesc, IntPtr x,
                    IntPtr wDesc, IntPtr w,
                    IntPtr convDesc,
                    ConvolutionFwdAlgo algo,
                    IntPtr workSpace,
                    Int64 workSpaceSizeInBytes,
                    IntPtr beta,
                    IntPtr yDesc, IntPtr y) {

                    Status status = NativeMethods.CudnnConvolutionForward.AsDelegate().Invoke(
                        handle, alpha, xDesc, x, wDesc, w, convDesc,
                        algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y
                    );
                    if (status != Status.Success) {
                        throw new CudaException(status);
                    }
                }
            }

            /// <summary>逆伝搬(特徴マップ)</summary>
            public static class BackwardData {

                /// <summary>アルゴリズム</summary>
                internal static ConvolutionBwdDataAlgo Algorithm(
                    IntPtr handle,
                    IntPtr wDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr dxDesc,
                    ConvolutionBwdDataPreference preference,
                    Int64 memoryLimitInBytes) {

                    ConvolutionBwdDataAlgo algo = ConvolutionBwdDataAlgo.Algo0;
                    
                    Status status = NativeMethods.CudnnGetConvolutionBackwardDataAlgorithm.AsDelegate().Invoke(
                        handle, wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes, ref algo
                    );
                    if (status != Status.Success) {
                        throw new CudaException(status);
                    }

                    return algo;
                }

                /// <summary>ワークスペースサイズ算出</summary>
                internal static Int64 WorkspaceSize(
                    IntPtr handle,
                    IntPtr wDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr dxDesc,
                    ConvolutionBwdDataAlgo algo) {

                    Int64 size = 0;

                    Status status = NativeMethods.CudnnGetConvolutionBackwardDataWorkspaceSize.AsDelegate().Invoke(
                        handle, wDesc, dyDesc, convDesc, dxDesc, algo, ref size
                    );
                    if (status != Status.Success) {
                        throw new CudaException(status);
                    }

                    return size;
                }

                /// <summary>実行</summary>
                internal static void Execute(
                    IntPtr handle,
                    IntPtr alpha,
                    IntPtr wDesc, IntPtr w,
                    IntPtr dyDesc, IntPtr dy,
                    IntPtr convDesc,
                    ConvolutionBwdDataAlgo algo,
                    IntPtr workSpace,
                    Int64 workSpaceSizeInBytes,
                    IntPtr beta,
                    IntPtr dxDesc, IntPtr dx) {

                    Status status = NativeMethods.CudnnConvolutionBackwardData.AsDelegate().Invoke(
                        handle, alpha, wDesc, w, dyDesc, dy, convDesc,
                        algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx
                    );
                    if (status != Status.Success) {
                        throw new CudaException(status);
                    }
                }
            }

            /// <summary>逆伝搬(フィルタ)</summary>
            public static class BackwardFilter {

                /// <summary>アルゴリズム</summary>
                internal static ConvolutionBwdFilterAlgo Algorithm(
                    IntPtr handle,
                    IntPtr xDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr dwDesc,
                    ConvolutionBwdFilterPreference preference,
                    Int64 memoryLimitInBytes) {

                    ConvolutionBwdFilterAlgo algo = ConvolutionBwdFilterAlgo.Algo0;
                    
                    Status status = NativeMethods.CudnnGetConvolutionBackwardFilterAlgorithm.AsDelegate().Invoke(
                        handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes, ref algo
                    );
                    if (status != Status.Success) {
                        throw new CudaException(status);
                    }

                    return algo;
                }

                /// <summary>ワークスペースサイズ算出</summary>
                internal static Int64 WorkspaceSize(
                    IntPtr handle,
                    IntPtr xDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr dwDesc,
                    ConvolutionBwdFilterAlgo algo) {

                    Int64 size = 0;

                    Status status = NativeMethods.CudnnGetConvolutionBackwardFilterWorkspaceSize.AsDelegate().Invoke(
                        handle, xDesc, dyDesc, convDesc, dwDesc, algo, ref size
                    );
                    if (status != Status.Success) {
                        throw new CudaException(status);
                    }

                    return size;
                }

                /// <summary>実行</summary>
                internal static void Execute(
                    IntPtr handle,
                    IntPtr alpha,
                    IntPtr xDesc, IntPtr x,
                    IntPtr dyDesc, IntPtr dy,
                    IntPtr convDesc,
                    ConvolutionBwdFilterAlgo algo,
                    IntPtr workSpace,
                    Int64 workSpaceSizeInBytes,
                    IntPtr beta,
                    IntPtr dwDesc, IntPtr dw) {

                    Status status = NativeMethods.CudnnConvolutionBackwardFilter.AsDelegate().Invoke(
                        handle, alpha, xDesc, x, dyDesc, dy, convDesc,
                        algo, workSpace,workSpaceSizeInBytes, beta, dwDesc, dw
                    );
                    if (status != Status.Success) {
                        throw new CudaException(status);
                    }
                }
            }
        }
    }
}
