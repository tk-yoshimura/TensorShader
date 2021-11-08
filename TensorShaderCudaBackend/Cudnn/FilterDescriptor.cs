using System;

namespace TensorShaderCudaBackend.Cudnn {

    /// <summary>フィルタパラメータ</summary>
    public class FilterDescriptor : IDisposable {
        private IntPtr ptr;

        internal IntPtr Ptr {
            get {
                if (ptr == IntPtr.Zero) {
                    throw new ObjectDisposedException(GetType().FullName);
                }

                return ptr;
            }
        }

        /// <summary>4Dフィルタ</summary>
        /// <param name="format">次元順</param>
        /// <param name="dtype">データ型</param>
        /// <param name="k">出力チャネル数</param>
        /// <param name="c">入力チャネル数</param>
        /// <param name="h">高さ</param>
        /// <param name="w">幅</param>
        public FilterDescriptor(TensorFormat format, DataType dtype, int k, int c, int h, int w) {
            ptr = API.Cudnn.TensorDescriptor.Create();
            API.Cudnn.FilterDescriptor.Set4d(ptr, format, dtype, k, c, h, w);
        }

        /// <summary>破棄</summary>
        public void Dispose() {
            API.Cudnn.TensorDescriptor.Destroy(ref ptr);
            GC.SuppressFinalize(this);
        }

        /// <summary>ファイナライザ</summary>
        ~FilterDescriptor() {
            Dispose();
        }
    }
}
