using System;

namespace TensorShaderCudaBackend.API {
    public static partial class Cudnn {

        /// <summary>ハンドル</summary>
        public static class Handle {

            /// <summary>作成</summary>
            internal static IntPtr Create() {
                if (!Exists) {
                    throw new CudaException("Cudnn library not found.");
                }

                IntPtr handle = IntPtr.Zero;
                Status status = Status.NotInitialized;

                if (Dll.CudaDll.CudnnVersion == 7) {
                    status = NativeMethods.Version7.CudnnCreate.AsDelegate().Invoke(ref handle);
                }
                else if (Dll.CudaDll.CudnnVersion == 8) {
                    status = NativeMethods.Version8.CudnnCreate.AsDelegate().Invoke(ref handle);
                }
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

                Status status = Status.NotInitialized;

                if (Dll.CudaDll.CudnnVersion == 7) {
                    status = NativeMethods.Version7.CudnnDestroy.AsDelegate().Invoke(handle);
                }
                else if (Dll.CudaDll.CudnnVersion == 8) {
                    status = NativeMethods.Version8.CudnnDestroy.AsDelegate().Invoke(handle);
                }

                if (status != Status.Success) {
                    throw new CudaException(status);
                }

                handle = IntPtr.Zero;
            }

            /// <summary>ストリーム割当</summary>
            internal static void SetStream(IntPtr handle, IntPtr stream) {
                Status status = Status.NotInitialized;

                if (Dll.CudaDll.CudnnVersion == 7) {
                    status = NativeMethods.Version7.CudnnSetStream.AsDelegate().Invoke(handle, stream);
                }
                else if (Dll.CudaDll.CudnnVersion == 8) {
                    status = NativeMethods.Version8.CudnnSetStream.AsDelegate().Invoke(handle, stream);
                }

                if (status != Status.Success) {
                    throw new CudaException(status);
                }
            }
        }
    }
}
