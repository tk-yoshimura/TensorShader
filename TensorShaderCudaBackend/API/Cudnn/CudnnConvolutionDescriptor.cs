using System;
using TensorShaderCudaBackend.Cudnn;

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
                    desc, pad.h, pad.w, stride.y, stride.x, dilation.y, dilation.x, TensorShaderCudaBackend.Cudnn.ConvolutionMode.CrossCorrelation, dtype
                );
                if (status != Status.Success) {
                    throw new CudaException(status);
                }
            }
        }
    }
}
