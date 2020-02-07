using System;
using TensorShaderCudaBackend.API;

namespace TensorShaderCudaBackend {

    /// <summary>ストリーム</summary>
    /// <remarks>カーネル実行キューに相当</remarks>
    public sealed class Stream : IDisposable {
        private IntPtr ptr;

        internal IntPtr Ptr {
            get {
                if (ptr == IntPtr.Zero) {
                    throw new ObjectDisposedException(GetType().FullName);
                }

                return ptr;
            }
        }

        /// <summary>有効か</summary>
        public bool IsValid => ptr != IntPtr.Zero;

        /// <summary>コンストラクタ</summary>
        public Stream() {
            this.ptr = Cuda.Stream.Create();
        }

        /// <summary>同期</summary>
        public void Synchronize() {
            Cuda.Stream.Synchronize(ptr);
        }

        /// <summary>文字列化</summary>
        public override string ToString() {
            return $"{nameof(Stream)}";
        }

        /// <summary>破棄</summary>
        public void Dispose() {
            Cuda.Stream.Destroy(ref ptr);

            GC.SuppressFinalize(this);

#if DEBUG
            Trace.WriteLine($"[{typeof(Stream).Name}.{MethodBase.GetCurrentMethod().Name}] Disposed stream");
#endif
        }

        /// <summary>ファイナライザ</summary>
        ~Stream() {
            Dispose();
        }
    }
}
