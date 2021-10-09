using System;
using System.Runtime.InteropServices;
using TensorShaderCudaBackend.Dll;

namespace TensorShaderCudaBackend.API {
    using CUfunc_cache = Nvcuda.FuncCache;
    using CUresult = Nvcuda.ResultCode;
    using size_t = Int64;

    public static partial class Nvcuda {

#pragma warning disable IDE1006
        private static class NativeMethods {

            static readonly NativeDll dll = CudaDll.Nvcuda;

            [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
            public delegate CUresult cuInit(uint Flags);
            public static NativeMethod<cuInit> CuInit { get; }
                = new NativeMethod<cuInit>(dll, nameof(cuInit));

            [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
            public delegate CUresult cuModuleLoadData(ref IntPtr module, IntPtr image);
            public static NativeMethod<cuModuleLoadData> CuModuleLoadData { get; }
                = new NativeMethod<cuModuleLoadData>(dll, nameof(cuModuleLoadData));

            [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
            public delegate CUresult cuModuleUnload(IntPtr hmod);
            public static NativeMethod<cuModuleUnload> CuModuleUnload { get; }
                = new NativeMethod<cuModuleUnload>(dll, nameof(cuModuleUnload));

            [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
            public delegate CUresult cuLaunchKernel(
                IntPtr kernel,
                uint gridDimX, uint gridDimY, uint gridDimZ,
                uint blockDimX, uint blockDimY, uint blockDimZ,
                uint sharedMemBytes, IntPtr hStream,
                IntPtr kernelParams,
                IntPtr extra
            );
            public static NativeMethod<cuLaunchKernel> CuLaunchKernel { get; }
                = new NativeMethod<cuLaunchKernel>(dll, nameof(cuLaunchKernel));

            [UnmanagedFunctionPointer(CallingConvention.Cdecl, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
            public delegate CUresult cuModuleGetFunction(ref IntPtr hfunc, IntPtr hmod, string name);
            public static NativeMethod<cuModuleGetFunction> CuModuleGetFunction { get; }
                = new NativeMethod<cuModuleGetFunction>(dll, nameof(cuModuleGetFunction));

            [UnmanagedFunctionPointer(CallingConvention.Cdecl, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
            public delegate CUresult cuModuleGetGlobal_v2(ref IntPtr dptr, ref size_t bytes, IntPtr hmod, string name);
            public static NativeMethod<cuModuleGetGlobal_v2> CuModuleGetGlobal_v2 { get; }
                = new NativeMethod<cuModuleGetGlobal_v2>(dll, nameof(cuModuleGetGlobal_v2));

            [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
            public delegate CUresult cuCtxSetCacheConfig(CUfunc_cache config);
            public static NativeMethod<cuCtxSetCacheConfig> CuCtxSetCacheConfig { get; }
                = new NativeMethod<cuCtxSetCacheConfig>(dll, nameof(cuCtxSetCacheConfig));

            [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
            public delegate CUresult cuGetErrorString(CUresult error, ref IntPtr pStr);
            public static NativeMethod<cuGetErrorString> CuGetErrorString { get; }
                = new NativeMethod<cuGetErrorString>(dll, nameof(cuGetErrorString));
        }
#pragma warning restore IDE1006
    }
}
