using System;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.API {
    public static partial class Cudnn {

        /// <summary>フィルタパラメータ</summary>
        public static class FilterDescriptor {

            /// <summary>作成</summary>
            internal static IntPtr Create() {
                if (!Exists) {
                    return IntPtr.Zero;
                }

                IntPtr desc = IntPtr.Zero;
                Status status = NativeMethods.CudnnCreateFilterDescriptor.AsDelegate().Invoke(ref desc);
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

                Status status = NativeMethods.CudnnDestroyFilterDescriptor.AsDelegate().Invoke(desc);
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
                int k, int c, int h, int w) {

                Status status = NativeMethods.CudnnSetFilter4dDescriptor.AsDelegate().Invoke(
                    desc, dtype, format, k, c, h, w
                );
                if (status != Status.Success) {
                    throw new CudaException(status);
                }
            }

            /// <summary>設定</summary>
            internal static void Set5d(
                IntPtr desc,
                TensorShaderCudaBackend.Cudnn.TensorFormat format,
                TensorShaderCudaBackend.Cudnn.DataType dtype,
                int k, int c, int d, int h, int w) {

                int[] dims = new int[] { k, c, d, h, w };

                GCHandle pinned_handle = GCHandle.Alloc(dims, GCHandleType.Pinned);
                IntPtr dims_ptr = pinned_handle.AddrOfPinnedObject();

                try {
                    Status status = NativeMethods.CudnnSetFilterNdDescriptor.AsDelegate().Invoke(
                        desc, dtype, format, 5, dims_ptr
                    );
                    if (status != Status.Success) {
                        throw new CudaException(status);
                    }
                }
                finally {
                    pinned_handle.Free();
                }
            }
        }
    }
}
