using System;
using System.Runtime.InteropServices;
using System.Text;

namespace TensorShaderCudaBackend.API {

    /// <summary>NVRTC API</summary>
    public static partial class Nvrtc {

        /// <summary>NVRTCバージョン</summary>
        public static (int major, int minor) Version {
            get {
                int major = 0, minor = 0;

                ResultCode result = NativeMethods.NvrtcVersion.AsDelegate().Invoke(ref major, ref minor);
                if (result != ResultCode.Success) {
                    throw new CudaException(result);
                }

                return (major, minor);
            }
        }

        /// <summary>シェーダプログラムのコンパイル</summary>
        internal static string CompileProgram(string code, string entrypoint, string[] options) {
            IntPtr prog = IntPtr.Zero;
            ResultCode result;

            code = $"extern \"C\" {{ {code} }}";

            result = NativeMethods.NvrtcCreateProgram.AsDelegate().Invoke(ref prog, code, entrypoint, 0, null, null);
            if (result != ResultCode.Success) {
                throw new CudaException(result);
            }

            try {
                result = NativeMethods.NvrtcCompileProgram.AsDelegate().Invoke(prog, options.Length, options);
                if (result != ResultCode.Success) {
                    long log_size = 0;

                    result = NativeMethods.NvrtcGetProgramLogSize.AsDelegate().Invoke(prog, ref log_size);
                    if (result != ResultCode.Success) {
                        throw new CudaException(result);
                    }

                    StringBuilder log = new((int)log_size + 8);
                    result = NativeMethods.NvrtcGetProgramLog.AsDelegate().Invoke(prog, log);
                    if (result != ResultCode.Success) {
                        throw new CudaException(result);
                    }

                    throw new CudaException($"Failure compile : {log}");
                }

                long ptx_size = 0;
                result = NativeMethods.NvrtcGetPTXSize.AsDelegate().Invoke(prog, ref ptx_size);
                if (result != ResultCode.Success) {
                    throw new CudaException(result);
                }

                StringBuilder ptx = new((int)ptx_size + 8);
                result = NativeMethods.NvrtcGetPTX.AsDelegate().Invoke(prog, ptx);
                if (result != ResultCode.Success) {
                    throw new CudaException(result);
                }

                return ptx.ToString();
            }
            catch (Exception) {
                throw;
            }
            finally {
                NativeMethods.NvrtcDestroyProgram.AsDelegate().Invoke(ref prog);
            }
        }

        /// <summary>エラーコードメッセージ</summary>
        internal static string GetErrorString(ResultCode result) {
            IntPtr ptr = NativeMethods.NvrtcGetErrorString.AsDelegate().Invoke(result);

            string str = string.Empty;
            if (ptr != IntPtr.Zero) {
                str = Marshal.PtrToStringAnsi(ptr);
            }

            return str;
        }
    }
}
