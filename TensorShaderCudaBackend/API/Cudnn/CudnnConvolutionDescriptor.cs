using System;

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
                Status status = Status.NotInitialized;

                if (Dll.CudaDll.CudnnVersion == 7) {
                    status = NativeMethods.Version7.CudnnCreateConvolutionDescriptor.AsDelegate().Invoke(ref desc);
                }
                else if (Dll.CudaDll.CudnnVersion == 8) {
                    status = NativeMethods.Version8.CudnnCreateConvolutionDescriptor.AsDelegate().Invoke(ref desc);
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
                    status = NativeMethods.Version7.CudnnDestroyConvolutionDescriptor.AsDelegate().Invoke(desc);
                }
                else if (Dll.CudaDll.CudnnVersion == 8) {
                    status = NativeMethods.Version8.CudnnDestroyConvolutionDescriptor.AsDelegate().Invoke(desc);
                }

                if (status != Status.Success) {
                    throw new CudaException(status);
                }

                desc = IntPtr.Zero;
            }

            /// <summary>設定</summary>
            internal static void SetConvolution2d(
                IntPtr desc, TensorShaderCudaBackend.Cudnn.DataType dtype,
                (int h, int w) pad, (int y, int x) stride, (int y, int x) dilation) {

                Status status = Status.NotInitialized;

                if (Dll.CudaDll.CudnnVersion == 7) {
                    status = NativeMethods.Version7.CudnnSetConvolution2dDescriptor.AsDelegate().Invoke(
                        desc, pad.h, pad.w, stride.y, stride.x, dilation.y, dilation.x, ConvolutionMode.CrossCorrelation, dtype
                    );
                }
                else if (Dll.CudaDll.CudnnVersion == 8) {
                    status = NativeMethods.Version8.CudnnSetConvolution2dDescriptor.AsDelegate().Invoke(
                        desc, pad.h, pad.w, stride.y, stride.x, dilation.y, dilation.x, ConvolutionMode.CrossCorrelation, dtype
                    );
                }

                if (status != Status.Success) {
                    throw new CudaException(status);
                }
            }
        }
    }
}
