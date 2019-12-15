using System;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.API {
    using CUresult = Nvcuda.ResultCode;
    using size_t = Int64;

    public static partial class Nvcuda {

        #pragma warning disable IDE1006 // 命名スタイル
        private static class NativeMethods {
            const string DllName = "nvcuda.dll";

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern CUresult cuInit(uint Flags);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern CUresult cuModuleLoadData(ref IntPtr module, IntPtr image);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern CUresult cuModuleUnload(IntPtr hmod);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern CUresult cuLaunchKernel(
                IntPtr kernel,
                uint gridDimX, uint gridDimY, uint gridDimZ,
                uint blockDimX, uint blockDimY, uint blockDimZ,
                uint sharedMemBytes, IntPtr hStream,
                IntPtr kernelParams,
                IntPtr extra
            );

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
            public static extern CUresult cuModuleGetFunction(ref IntPtr hfunc, IntPtr hmod, string name);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
            public static extern CUresult cuModuleGetGlobal_v2(ref IntPtr dptr, ref size_t bytes, IntPtr hmod, string name);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern CUresult cuGetErrorString(CUresult error, ref IntPtr pStr);
        }
        #pragma warning restore IDE1006 // 命名スタイル
    }
}
