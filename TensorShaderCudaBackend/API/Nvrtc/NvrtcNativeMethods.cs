using System;
using System.Runtime.InteropServices;
using System.Text;
using TensorShaderCudaBackend.Dll;

namespace TensorShaderCudaBackend.API {
    using nvrtcResult = Nvrtc.ResultCode;

    public static partial class Nvrtc {

#pragma warning disable IDE1006
        private static class NativeMethods {

            static readonly NativeDll dll = CudaDll.Nvrtc;

            [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
            public delegate nvrtcResult nvrtcVersion(ref int major, ref int minor);
            public static NativeMethod<nvrtcVersion> NvrtcVersion { get; }
                = new NativeMethod<nvrtcVersion>(dll, nameof(nvrtcVersion));

            [UnmanagedFunctionPointer(CallingConvention.Cdecl, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
            public delegate nvrtcResult nvrtcCompileProgram(
                IntPtr prog,
                int numOptions,
                [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr, SizeParamIndex = 1)]
                string[] options
            );
            public static NativeMethod<nvrtcCompileProgram> NvrtcCompileProgram { get; }
                = new NativeMethod<nvrtcCompileProgram>(dll, nameof(nvrtcCompileProgram));

            [UnmanagedFunctionPointer(CallingConvention.Cdecl, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
            public delegate nvrtcResult nvrtcCreateProgram(
                ref IntPtr prog,
                string src,
                string name,
                int numHeaders,
                [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr, SizeParamIndex = 1)]
                string[] headers,
                [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr, SizeParamIndex = 1)]
                string[] includeNames
            );
            public static NativeMethod<nvrtcCreateProgram> NvrtcCreateProgram { get; }
                = new NativeMethod<nvrtcCreateProgram>(dll, nameof(nvrtcCreateProgram));

            [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
            public delegate nvrtcResult nvrtcDestroyProgram(ref IntPtr prog);
            public static NativeMethod<nvrtcDestroyProgram> NvrtcDestroyProgram { get; }
                = new NativeMethod<nvrtcDestroyProgram>(dll, nameof(nvrtcDestroyProgram));

            [UnmanagedFunctionPointer(CallingConvention.Cdecl, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
            public delegate nvrtcResult nvrtcGetPTX(IntPtr prog, StringBuilder ptx);
            public static NativeMethod<nvrtcGetPTX> NvrtcGetPTX { get; }
                = new NativeMethod<nvrtcGetPTX>(dll, nameof(nvrtcGetPTX));

            [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
            public delegate nvrtcResult nvrtcGetPTXSize(IntPtr prog, ref long ptxSizeRet);
            public static NativeMethod<nvrtcGetPTXSize> NvrtcGetPTXSize { get; }
                = new NativeMethod<nvrtcGetPTXSize>(dll, nameof(nvrtcGetPTXSize));

            [UnmanagedFunctionPointer(CallingConvention.Cdecl, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
            public delegate nvrtcResult nvrtcGetProgramLog(IntPtr prog, StringBuilder log);
            public static NativeMethod<nvrtcGetProgramLog> NvrtcGetProgramLog { get; }
                = new NativeMethod<nvrtcGetProgramLog>(dll, nameof(nvrtcGetProgramLog));

            [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
            public delegate nvrtcResult nvrtcGetProgramLogSize(IntPtr prog, ref long logSizeRet);
            public static NativeMethod<nvrtcGetProgramLogSize> NvrtcGetProgramLogSize { get; }
                = new NativeMethod<nvrtcGetProgramLogSize>(dll, nameof(nvrtcGetProgramLogSize));

            [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
            public delegate IntPtr nvrtcGetErrorString(nvrtcResult result);
            public static NativeMethod<nvrtcGetErrorString> NvrtcGetErrorString { get; }
                = new NativeMethod<nvrtcGetErrorString>(dll, nameof(nvrtcGetErrorString));
        }
#pragma warning restore IDE1006
    }
}
