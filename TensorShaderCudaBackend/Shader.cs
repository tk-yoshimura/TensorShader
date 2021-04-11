namespace TensorShaderCudaBackend {

    /// <summary>コンピュートシェーダー</summary>
    public abstract partial class Shader {

        private static Stream stream = null;

        /// <summary>カーネル</summary>
        protected internal Kernel Kernel { set; get; }

        /// <summary>実行</summary>
        public abstract void Execute(Stream stream, params object[] args);

        /// <summary>引数チェック</summary>
        protected abstract void CheckArgument(params object[] args);

        /// <summary>概要</summary>
        public string Overview => Kernel.Overview;

        /// <summary>識別子</summary>
        public virtual string Signature { get; }

        /// <summary>デフォルトストリーム</summary>
        public static Stream DefaultStream {
            get {
                if (stream is null) {
                    stream = new Stream();
                }

                return stream;
            }
        }

        /// <summary>文字列化</summary>
        public override string ToString() {
            return Signature;
        }
    }
}
