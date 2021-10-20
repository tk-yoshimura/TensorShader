using System;

namespace TensorShaderCudaBackend.API {
    public static partial class Cudnn {

        /// <summary>ハンドル</summary>
        public static class Handle {

            /// <summary>作成</summary>
            internal static IntPtr Create() {
                if (!Exists) {
                    return IntPtr.Zero;
                }

                IntPtr handle = IntPtr.Zero;
                Status status = NativeMethods.CudnnCreate.AsDelegate().Invoke(ref handle);
                if (status != Status.Success) {
                    throw new CudaException(status);
                }

                return handle;
            }

            /// <summary>破棄</summary>
            internal static void Destroy(ref IntPtr handle) {
                if (handle == IntPtr.Zero) {
                    return;
                }

                Status status = NativeMethods.CudnnDestroy.AsDelegate().Invoke(handle);
                if (status != Status.Success) {
                    throw new CudaException(status);
                }

                handle = IntPtr.Zero;
            }

            /// <summary>ストリーム割当</summary>
            internal static void SetStream(IntPtr handle, IntPtr stream) {
                Status status = NativeMethods.CudnnSetStream.AsDelegate().Invoke(handle, stream);
                if (status != Status.Success) {
                    throw new CudaException(status);
                }
            }
        }
    }
}
