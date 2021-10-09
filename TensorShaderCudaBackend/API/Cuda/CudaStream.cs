using System;

namespace TensorShaderCudaBackend.API {
    public static partial class Cuda {

        /// <summary>ストリーム操作</summary>
        public static class Stream {

            /// <summary>ストリーム作成</summary>
            internal static IntPtr Create() {
                IntPtr stream = IntPtr.Zero;
                ErrorCode result = NativeMethods.CudaStreamCreate.AsDelegate().Invoke(ref stream);
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }

                return stream;
            }

            /// <summary>ストリーム破棄</summary>
            internal static void Destroy(ref IntPtr stream) {
                if (stream == IntPtr.Zero) {
                    return;
                }

                ErrorCode result = NativeMethods.CudaStreamDestroy.AsDelegate().Invoke(stream);
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }

                stream = IntPtr.Zero;
            }

            /// <summary>同期</summary>
            internal static void Synchronize(IntPtr stream) {
                ErrorCode result = NativeMethods.CudaStreamSynchronize.AsDelegate().Invoke(stream);
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }
            }
        }
    }
}
