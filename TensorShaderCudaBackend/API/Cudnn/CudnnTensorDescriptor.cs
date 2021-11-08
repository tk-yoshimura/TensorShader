using System;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.API {
    public static partial class Cudnn {

        /// <summary>テンソルパラメータ</summary>
        public static class TensorDescriptor {

            /// <summary>作成</summary>
            internal static IntPtr Create() {
                if (!Exists) {
                    return IntPtr.Zero;
                }

                IntPtr desc = IntPtr.Zero;
                Status status = Status.NotInitialized;

                if (Dll.CudaDll.CudnnVersion == 7) {
                    status = NativeMethods.Version7.CudnnCreateTensorDescriptor.AsDelegate().Invoke(ref desc);
                }
                else if (Dll.CudaDll.CudnnVersion == 8) {
                    status = NativeMethods.Version8.CudnnCreateTensorDescriptor.AsDelegate().Invoke(ref desc);
                }

                if (status != Status.Success) {
                    throw new CudaException(status);
                }

                return desc;
            }

            /// <summary>破棄</summary>
            internal static void Destroy(ref IntPtr desc) {
                if (desc == IntPtr.Zero) {
                    return;
                }

                Status status = Status.NotInitialized;

                if (Dll.CudaDll.CudnnVersion == 7) {
                    status = NativeMethods.Version7.CudnnDestroyTensorDescriptor.AsDelegate().Invoke(desc);
                }
                else if (Dll.CudaDll.CudnnVersion == 8) {
                    status = NativeMethods.Version8.CudnnDestroyTensorDescriptor.AsDelegate().Invoke(desc);
                }

                if (status != Status.Success) {
                    throw new CudaException(status);
                }

                desc = IntPtr.Zero;
            }

            /// <summary>設定</summary>
            internal static void Set4d(
                IntPtr desc,
                TensorShaderCudaBackend.Cudnn.TensorFormat format,
                TensorShaderCudaBackend.Cudnn.DataType dtype,
                int n, int c, int h, int w) {

                Status status = Status.NotInitialized;

                if (Dll.CudaDll.CudnnVersion == 7) {
                    status = NativeMethods.Version7.CudnnSetTensor4dDescriptor.AsDelegate().Invoke(
                        desc, format, dtype, n, c, h, w
                    );
                }
                else if (Dll.CudaDll.CudnnVersion == 8) {
                    status = NativeMethods.Version8.CudnnSetTensor4dDescriptor.AsDelegate().Invoke(
                        desc, format, dtype, n, c, h, w
                    );
                }

                if (status != Status.Success) {
                    throw new CudaException(status);
                }
            }
        }
    }
}
