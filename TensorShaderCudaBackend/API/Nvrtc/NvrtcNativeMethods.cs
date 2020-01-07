using System;
using System.Runtime.InteropServices;
using System.Text;

namespace TensorShaderCudaBackend.API {
    using nvrtcResult = Nvrtc.ResultCode;

    public static partial class Nvrtc {

#pragma warning disable IDE1006 // 命名スタイル
        private static class NativeMethods {
#if CUDA_10_0
            const string DllName = "nvrtc64_100_0.dll";
#elif CUDA_10_1
            const string DllName = "nvrtc64_101_0.dll";
#elif CUDA_10_2
            const string DllName = "nvrtc64_102_0.dll";
#elif CUDA_10_3
            const string DllName = "nvrtc64_103_0.dll";
#elif CUDA_10_4
            const string DllName = "nvrtc64_104_0.dll";
#else
            const string DllName = "nvrtc64_101_0.dll";
#endif

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern nvrtcResult nvrtcVersion(ref int major, ref int minor);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
            public static extern nvrtcResult nvrtcCompileProgram(
                IntPtr prog,
                int numOptions,
                [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr, SizeParamIndex = 1)]
                string[] options
            );

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
            public static extern nvrtcResult nvrtcCreateProgram(
                ref IntPtr prog,
                string src,
                string name,
                int numHeaders,
                [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr, SizeParamIndex = 1)]
                string[] headers,
                [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr, SizeParamIndex = 1)]
                string[] includeNames
            );

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern nvrtcResult nvrtcDestroyProgram(ref IntPtr prog);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
            public static extern nvrtcResult nvrtcGetPTX(IntPtr prog, StringBuilder ptx);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern nvrtcResult nvrtcGetPTXSize(IntPtr prog, ref long ptxSizeRet);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
            public static extern nvrtcResult nvrtcGetProgramLog(IntPtr prog, StringBuilder log);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern nvrtcResult nvrtcGetProgramLogSize(IntPtr prog, ref long logSizeRet);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern IntPtr nvrtcGetErrorString(nvrtcResult result);
        }
#pragma warning restore IDE1006 // 命名スタイル
    }
}
