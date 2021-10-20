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
                Status status = NativeMethods.CudnnCreateTensorDescriptor.AsDelegate().Invoke(ref desc);
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

                Status status = NativeMethods.CudnnDestroyTensorDescriptor.AsDelegate().Invoke(desc);
                if (status != Status.Success) {
                    throw new CudaException(status);
                }

                desc = IntPtr.Zero;
            }

            /// <summary>設定</summary>
            internal static void Set4d(IntPtr desc, TensorFormat format, DataType dtype, int n, int c, int h, int w) {
                Status status = NativeMethods.CudnnSetTensor4dDescriptor.AsDelegate().Invoke(
                    desc, format, dtype, n, c, h, w
                );
                if (status != Status.Success) {
                    throw new CudaException(status);
                }
            }

            /// <summary>設定</summary>
            internal static void Set5d(IntPtr desc, TensorFormat format, DataType dtype, int n, int c, int d, int h, int w) {
                int[] dims = new int[] { n, c, d, h, w };
                
                GCHandle pinned_handle = GCHandle.Alloc(dims, GCHandleType.Pinned);
                IntPtr dims_ptr = GCHandle.ToIntPtr(pinned_handle);

                try {
                    Status status = NativeMethods.CudnnSetTensorNdDescriptorEx.AsDelegate().Invoke(
                        desc, format, dtype, 5, dims_ptr
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
