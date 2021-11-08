using System;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.API {
    /// <summary>Cudnn API</summary>
    public static partial class Cudnn {

        /// <summary>CUDNNライブラリが実行環境に存在するか</summary>
        public static bool Exists => Dll.CudaDll.Cudnn is not null;

        /// <summary>エラーコードメッセージ</summary>
        internal static string GetErrorString(Status status) {
            if (!Exists) {
                return "CUDNN not exists.";
            }

            IntPtr ptr = IntPtr.Zero;

            if (Dll.CudaDll.CudnnVersion == 7) {
                ptr = NativeMethods.Version7.CudnnGetErrorString.AsDelegate().Invoke(status);
            }
            else if (Dll.CudaDll.CudnnVersion == 8) {
                ptr = NativeMethods.Version8.CudnnGetErrorString.AsDelegate().Invoke(status);
            }

            string str = string.Empty;
            if (ptr != IntPtr.Zero) {
                str = Marshal.PtrToStringAnsi(ptr);
            }

            return str;
        }
    }
}
