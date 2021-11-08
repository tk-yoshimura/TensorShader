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
                    IntPtr xDesc, IntPtr x, IntPtr wDesc, IntPtr w, IntPtr convDesc, IntPtr yDesc, IntPtr y,
                    ConvolutionFwdPreference preference, IntPtr memory, Int64 memoryLimitInBytes) {

                    ConvolutionFwdAlgo algo = ConvolutionFwdAlgo.ImplicitGemm;
                    Status status = Status.NotInitialized;

                    if (Dll.CudaDll.CudnnVersion == 7) {
                        status = NativeMethods.Version7.CudnnGetConvolutionForwardAlgorithm.AsDelegate().Invoke(
                            handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, ref algo
                        );
                    }
                    else if (Dll.CudaDll.CudnnVersion == 8){
                        ConvolutionFwdAlgoPerf[] prefs = new ConvolutionFwdAlgoPerf[1];
                        GCHandle pinned_handle = GCHandle.Alloc(prefs, GCHandleType.Pinned);
                        int count = 0;
                        
                        try {
                            IntPtr prefs_ptr = pinned_handle.AddrOfPinnedObject();

                            status = NativeMethods.Version8.CudnnFindConvolutionForwardAlgorithmEx.AsDelegate().Invoke(
                                handle, xDesc, x, wDesc, w, convDesc, yDesc, y, 1, ref count, prefs_ptr, memory, memoryLimitInBytes
                            );
                        }
                        finally {
                            pinned_handle.Free();
                        }

                        if (status == Status.Success && count >= 1) {
                            algo = prefs[0].algo;
                        }
                    }

                    if (status != Status.Success) {
                        throw new CudaException(status);
                    }

                    return algo;
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

                    Status status = Status.NotInitialized;

                    if (Dll.CudaDll.CudnnVersion == 7) {
                        status = NativeMethods.Version7.CudnnConvolutionForward.AsDelegate().Invoke(
                            handle, alpha, xDesc, x, wDesc, w, convDesc,
                            algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y
                        );
                    }
                    else if (Dll.CudaDll.CudnnVersion == 8) {
                        status = NativeMethods.Version8.CudnnConvolutionForward.AsDelegate().Invoke(
                            handle, alpha, xDesc, x, wDesc, w, convDesc,
                            algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y
                        );
                    }

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
                    IntPtr wDesc, IntPtr w, IntPtr dyDesc, IntPtr dy, IntPtr convDesc, IntPtr dxDesc, IntPtr dx,
                    ConvolutionBwdDataPreference preference,
                    IntPtr memory, Int64 memoryLimitInBytes) {

                    ConvolutionBwdDataAlgo algo = ConvolutionBwdDataAlgo.Algo0;
                    Status status = Status.NotInitialized;

                    if (Dll.CudaDll.CudnnVersion == 7) {
                        status = NativeMethods.Version7.CudnnGetConvolutionBackwardDataAlgorithm.AsDelegate().Invoke(
                            handle, wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes, ref algo
                        );
                    }
                    else if (Dll.CudaDll.CudnnVersion == 8) {
                        ConvolutionBwdDataAlgoPerf[] prefs = new ConvolutionBwdDataAlgoPerf[1];
                        GCHandle pinned_handle = GCHandle.Alloc(prefs, GCHandleType.Pinned);
                        int count = 0;

                        try {
                            IntPtr prefs_ptr = pinned_handle.AddrOfPinnedObject();

                            status = NativeMethods.Version8.CudnnFindConvolutionBackwardDataAlgorithmEx.AsDelegate().Invoke(
                                handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, 1, ref count, prefs_ptr, memory, memoryLimitInBytes
                            );
                        }
                        finally {
                            pinned_handle.Free();
                        }

                        if (status == Status.Success && count >= 1) {
                            algo = prefs[0].algo;
                        }
                    }

                    if (status != Status.Success) {
                        throw new CudaException(status);
                    }

                    return algo;
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

                    Status status = Status.NotInitialized;

                    if (Dll.CudaDll.CudnnVersion == 7) {
                        status = NativeMethods.Version7.CudnnConvolutionBackwardData.AsDelegate().Invoke(
                            handle, alpha, wDesc, w, dyDesc, dy, convDesc,
                            algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx
                        );
                    }
                    else if (Dll.CudaDll.CudnnVersion == 8) {
                        status = NativeMethods.Version8.CudnnConvolutionBackwardData.AsDelegate().Invoke(
                            handle, alpha, wDesc, w, dyDesc, dy, convDesc,
                            algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx
                        );
                    }

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
                    IntPtr xDesc, IntPtr x, IntPtr dyDesc, IntPtr dy, IntPtr convDesc, IntPtr dwDesc, IntPtr dw,
                    ConvolutionBwdFilterPreference preference,
                    IntPtr memory, Int64 memoryLimitInBytes) {

                    ConvolutionBwdFilterAlgo algo = ConvolutionBwdFilterAlgo.Algo0;
                    Status status = Status.NotInitialized;

                    if (Dll.CudaDll.CudnnVersion == 7) {
                        status = NativeMethods.Version7.CudnnGetConvolutionBackwardFilterAlgorithm.AsDelegate().Invoke(
                            handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes, ref algo
                        );
                    }
                    else if (Dll.CudaDll.CudnnVersion == 8) {
                        ConvolutionBwdFilterAlgoPerf[] prefs = new ConvolutionBwdFilterAlgoPerf[1];
                        GCHandle pinned_handle = GCHandle.Alloc(prefs, GCHandleType.Pinned);
                        int count = 0;

                        try {
                            IntPtr prefs_ptr = pinned_handle.AddrOfPinnedObject();

                            status = NativeMethods.Version8.CudnnFindConvolutionBackwardFilterAlgorithmEx.AsDelegate().Invoke(
                                handle, xDesc, x, dyDesc, dy, convDesc, dwDesc, dw, 1, ref count, prefs_ptr, memory, memoryLimitInBytes
                            );
                        }
                        finally {
                            pinned_handle.Free();
                        }

                        if (status == Status.Success && count >= 1) {
                            algo = prefs[0].algo;
                        }
                    }

                    if (status != Status.Success) {
                        throw new CudaException(status);
                    }

                    return algo;
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

                    Status status = Status.NotInitialized;

                    if (Dll.CudaDll.CudnnVersion == 7) {
                        status = NativeMethods.Version7.CudnnConvolutionBackwardFilter.AsDelegate().Invoke(
                            handle, alpha, xDesc, x, dyDesc, dy, convDesc,
                            algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw
                        );
                    }
                    else if (Dll.CudaDll.CudnnVersion == 8) {
                        status = NativeMethods.Version8.CudnnConvolutionBackwardFilter.AsDelegate().Invoke(
                            handle, alpha, xDesc, x, dyDesc, dy, convDesc,
                            algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw
                        );
                    }

                    if (status != Status.Success) {
                        throw new CudaException(status);
                    }
                }
            }
        }
    }
}
