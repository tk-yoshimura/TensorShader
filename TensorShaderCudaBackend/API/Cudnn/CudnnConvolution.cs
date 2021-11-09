using System;
using System.Linq;
using System.Runtime.InteropServices;
using TensorShaderCudaBackend.Cudnn;

namespace TensorShaderCudaBackend.API {
    public static partial class Cudnn {

        /// <summary>畳み込み</summary>
        public static class Convolution {

            /// <summary>順伝搬</summary>
            public static class Forward {

                /// <summary>アルゴリズム</summary>
                internal static ConvolutionFwdAlgoPerf[] EnumAlgorithm(
                    IntPtr handle,
                    IntPtr xDesc, IntPtr wDesc, IntPtr convDesc, IntPtr yDesc, int algo_requests, Int64 workspace_limit_bytes) {

                    ConvolutionFwdAlgoPerf[] prefs = new ConvolutionFwdAlgoPerf[algo_requests];
                    GCHandle pinned_handle = GCHandle.Alloc(prefs, GCHandleType.Pinned);
                    int count = 0;

                    try {
                        IntPtr prefs_ptr = pinned_handle.AddrOfPinnedObject();

                        Status status = NativeMethods.CudnnGetConvolutionForwardAlgorithm.AsDelegate().Invoke(
                            handle, xDesc, wDesc, convDesc, yDesc, algo_requests, ref count, prefs_ptr
                        );
                        if (status != Status.Success) {
                            throw new CudaException(status);
                        }
                    }
                    finally {
                        pinned_handle.Free();
                    }

                    prefs = prefs
                        .Take(count)
                        .Where((pref) => pref.status == Status.Success && pref.memory <= workspace_limit_bytes)
                        .ToArray();

                    return prefs;
                }

                /// <summary>ワークスペースサイズ算出</summary>
                internal static Int64 GetWorkspaceSize(
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
                internal static ConvolutionBwdDataAlgoPerf[] EnumAlgorithm(
                    IntPtr handle,
                    IntPtr wDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr dxDesc, int algo_requests, Int64 workspace_limit_bytes) {

                    ConvolutionBwdDataAlgoPerf[] prefs = new ConvolutionBwdDataAlgoPerf[algo_requests];
                    GCHandle pinned_handle = GCHandle.Alloc(prefs, GCHandleType.Pinned);
                    int count = 0;

                    try {
                        IntPtr prefs_ptr = pinned_handle.AddrOfPinnedObject();

                        Status status = NativeMethods.CudnnGetConvolutionBackwardDataAlgorithm.AsDelegate().Invoke(
                            handle, wDesc, dyDesc, convDesc, dxDesc, algo_requests, ref count, prefs_ptr
                        );
                        if (status != Status.Success) {
                            throw new CudaException(status);
                        }
                    }
                    finally {
                        pinned_handle.Free();
                    }

                    prefs = prefs
                        .Take(count)
                        .Where((pref) => pref.status == Status.Success && pref.memory <= workspace_limit_bytes)
                        .ToArray();

                    return prefs;
                }

                /// <summary>ワークスペースサイズ算出</summary>
                internal static Int64 GetWorkspaceSize(
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
                internal static ConvolutionBwdFilterAlgoPerf[] EnumAlgorithm(
                    IntPtr handle,
                    IntPtr xDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr dwDesc, int algo_requests, Int64 workspace_limit_bytes) {

                    ConvolutionBwdFilterAlgoPerf[] prefs = new ConvolutionBwdFilterAlgoPerf[algo_requests];
                    GCHandle pinned_handle = GCHandle.Alloc(prefs, GCHandleType.Pinned);
                    int count = 0;

                    try {
                        IntPtr prefs_ptr = pinned_handle.AddrOfPinnedObject();

                        Status status = NativeMethods.CudnnGetConvolutionBackwardFilterAlgorithm.AsDelegate().Invoke(
                            handle, xDesc, dyDesc, convDesc, dwDesc, algo_requests, ref count, prefs_ptr
                        );
                        if (status != Status.Success) {
                            throw new CudaException(status);
                        }
                    }
                    finally {
                        pinned_handle.Free();
                    }

                    prefs = prefs
                        .Take(count)
                        .Where((pref) => pref.status == Status.Success && pref.memory <= workspace_limit_bytes)
                        .ToArray();

                    return prefs;
                }

                /// <summary>ワークスペースサイズ算出</summary>
                internal static Int64 GetWorkspaceSize(
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
                        algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw
                    );
                    if (status != Status.Success) {
                        throw new CudaException(status);
                    }
                }
            }
        }
    }
}
