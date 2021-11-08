using System;

namespace TensorShaderCudaBackend.Cudnn {

    /// <summary>畳み込みパラメータ</summary>
    public class ConvolutionDescriptor : IDisposable {
        private IntPtr ptr;

        internal IntPtr Ptr {
            get {
                if (ptr == IntPtr.Zero) {
                    throw new ObjectDisposedException(GetType().FullName);
                }

                return ptr;
            }
        }

        /// <summary>2D畳み込み</summary>
        public ConvolutionDescriptor(
            DataType dtype,
            (int h, int w) pad, (int y, int x) stride, (int y, int x) dilation) {

            ptr = API.Cudnn.ConvolutionDescriptor.Create();
            API.Cudnn.ConvolutionDescriptor.SetConvolution2d(ptr, dtype, pad, stride, dilation);
        }

        /// <summary>破棄</summary>
        public void Dispose() {
            API.Cudnn.ConvolutionDescriptor.Destroy(ref ptr);
            GC.SuppressFinalize(this);
        }

        /// <summary>ファイナライザ</summary>
        ~ConvolutionDescriptor() {
            Dispose();
        }
    }
}
