using System;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.API {
    public static partial class Cudnn {

        /// <summary>畳み込みパラメータ</summary>
        public static class ConvolutionDescriptor {

            /// <summary>作成</summary>
            internal static IntPtr Create() {
                if (!Exists) {
                    return IntPtr.Zero;
                }

                IntPtr desc = IntPtr.Zero;
                Status status = NativeMethods.CudnnCreateConvolutionDescriptor.AsDelegate().Invoke(ref desc);
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

                Status status = NativeMethods.CudnnDestroyConvolutionDescriptor.AsDelegate().Invoke(desc);
                if (status != Status.Success) {
                    throw new CudaException(status);
                }

                desc = IntPtr.Zero;
            }

            /// <summary>設定</summary>
            internal static void SetConvolution2d(
                IntPtr desc, TensorShaderCudaBackend.Cudnn.DataType dtype, 
                (int h, int w) pad, (int y, int x) stride, (int y, int x) dilation) {

                Status status = NativeMethods.CudnnSetConvolution2dDescriptor.AsDelegate().Invoke(
                    desc, pad.h, pad.w, stride.y, stride.x, dilation.y, dilation.x, ConvolutionMode.CrossCorrelation, dtype
                );
                if (status != Status.Success) {
                    throw new CudaException(status);
                }
            }

            /// <summary>設定</summary>
            internal static void SetConvolution3d(
                IntPtr desc, TensorShaderCudaBackend.Cudnn.DataType dtype, 
                (int d, int h, int w) pad, (int z, int y, int x) stride, (int z, int y, int x) dilation) {

                int[] pads = new int[] { pad.d, pad.h, pad.w };
                int[] strides = new int[] { stride.z, stride.y, stride.x };
                int[] dilations = new int[] { dilation.z, dilation.y, dilation.x };
                
                GCHandle pinned_pads_handle = GCHandle.Alloc(pads, GCHandleType.Pinned);
                GCHandle pinned_strides_handle = GCHandle.Alloc(strides, GCHandleType.Pinned);
                GCHandle pinned_dilations_handle = GCHandle.Alloc(dilations, GCHandleType.Pinned);

                IntPtr pads_ptr = pinned_pads_handle.AddrOfPinnedObject();
                IntPtr strides_ptr = pinned_strides_handle.AddrOfPinnedObject();
                IntPtr dilations_ptr = pinned_dilations_handle.AddrOfPinnedObject();

                try {
                    Status status = NativeMethods.CudnnSetConvolutionNdDescriptor.AsDelegate().Invoke(
                        desc, 3, pads_ptr, strides_ptr, dilations_ptr, ConvolutionMode.CrossCorrelation, dtype 
                    );
                    if (status != Status.Success) {
                        throw new CudaException(status);
                    }
                }
                finally {
                    pinned_pads_handle.Free();
                    pinned_strides_handle.Free();
                    pinned_dilations_handle.Free();
                }
            }
        }
    }
}
