using System;
using System.Text;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.API {

    /// <summary>NVRTC API</summary>
    public static partial class Nvrtc {

        /// <summary>NVRTCバージョン</summary>
        public static (int major, int minor) Version {
            get {
                int major = 0, minor = 0;

                ResultCode result = NativeMethods.nvrtcVersion(ref major, ref minor);
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

            result = NativeMethods.nvrtcCreateProgram(ref prog, code, entrypoint, 0, null, null);
            if (result != ResultCode.Success) {
                throw new CudaException(result);
            }

            try {
                result = NativeMethods.nvrtcCompileProgram(prog, options.Length, options);
                if (result != ResultCode.Success) {
                    long log_size = 0;

                    result = NativeMethods.nvrtcGetProgramLogSize(prog, ref log_size);
                    if (result != ResultCode.Success) {
                        throw new CudaException(result);
                    }

                    StringBuilder log = new StringBuilder((int)log_size + 8);
                    result = NativeMethods.nvrtcGetProgramLog(prog, log);
                    if (result != ResultCode.Success) {
                        throw new CudaException(result);
                    }

                    throw new CudaException($"Failure compile : {log.ToString()}");
                }

                long ptx_size = 0;
                result = NativeMethods.nvrtcGetPTXSize(prog, ref ptx_size);
                if (result != ResultCode.Success) {
                    throw new CudaException(result);
                }

                StringBuilder ptx = new StringBuilder((int)ptx_size + 8);
                result = NativeMethods.nvrtcGetPTX(prog, ptx);
                if (result != ResultCode.Success) {
                    throw new CudaException(result);
                }

                return ptx.ToString();
            }
            catch (Exception) {
                throw;
            }
            finally {
                NativeMethods.nvrtcDestroyProgram(ref prog);
            }
        }

        /// <summary>エラーコードメッセージ</summary>
        internal static string GetErrorString(ResultCode result) {
            IntPtr ptr = NativeMethods.nvrtcGetErrorString(result);
            return Marshal.PtrToStringAnsi(ptr);
        }
    }
}
