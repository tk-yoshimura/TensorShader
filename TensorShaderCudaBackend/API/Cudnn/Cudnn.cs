using System;
using System.Runtime.InteropServices;
using TensorShaderCudaBackend.Cudnn;

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

            IntPtr ptr = NativeMethods.CudnnGetErrorString.AsDelegate().Invoke(status);

            string str = string.Empty;
            if (ptr != IntPtr.Zero) {
                str = Marshal.PtrToStringAnsi(ptr);
            }

            return str;
        }
    }
}
