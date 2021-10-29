using System;

namespace TensorShaderCudaBackend.Cudnn {

    /// <summary>テンソルパラメータ</summary>
    public class TensorDescriptor : IDisposable {
        private IntPtr ptr;

        internal IntPtr Ptr {
            get {
                if (ptr == IntPtr.Zero) {
                    throw new ObjectDisposedException(GetType().FullName);
                }

                return ptr;
            }
        }

        /// <summary>4Dテンソル</summary>
        /// <param name="format">次元順</param>
        /// <param name="dtype">データ型</param>
        /// <param name="n">バッチサイズ</param>
        /// <param name="c">チャネル数</param>
        /// <param name="h">高さ</param>
        /// <param name="w">幅</param>
        public TensorDescriptor(TensorFormat format, DataType dtype, int n, int c, int h, int w) {
            ptr = API.Cudnn.TensorDescriptor.Create();
            API.Cudnn.TensorDescriptor.Set4d(ptr, format, dtype, n, c, h, w);
        }

        /// <summary>5Dテンソル</summary>
        /// <param name="format">次元順</param>
        /// <param name="dtype">データ型</param>
        /// <param name="n">バッチサイズ</param>
        /// <param name="c">チャネル数</param>
        /// <param name="d">奥行き</param>
        /// <param name="h">高さ</param>
        /// <param name="w">幅</param>
        public TensorDescriptor(TensorFormat format, DataType dtype, int n, int c, int d, int h, int w) {
            ptr = API.Cudnn.TensorDescriptor.Create();
            API.Cudnn.TensorDescriptor.Set5d(ptr, format, dtype, n, c, d, h, w);
        }

        /// <summary>破棄</summary>
        public void Dispose() {
            API.Cudnn.TensorDescriptor.Destroy(ref ptr);
            GC.SuppressFinalize(this);
        }

        /// <summary>ファイナライザ</summary>
        ~TensorDescriptor() {
            Dispose();
        }
    }
}
