namespace TensorShaderCudaBackend {

    /// <summary>コンピュートシェーダー</summary>
    public abstract class Shader {

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
                if (stream == null) {
                    stream = new Stream();
                }

                return stream;
            }
        }

        /// <summary>文字列化</summary>
        public override string ToString() {
            return Signature;
        }

        /// <summary>定義済み変数</summary>
        protected static class Defines {
            /// <summary>Xインデクス</summary>
            public static string IndexX => "(blockDim.x * blockIdx.x + threadIdx.x)";

            /// <summary>Yインデクス</summary>
            public static string IndexY => "(blockDim.y * blockIdx.y + threadIdx.y)";

            /// <summary>Zインデクス</summary>
            public static string IndexZ => "(blockDim.z * blockIdx.z + threadIdx.z)";

            /// <summary>Xブロックインデクス</summary>
            public static string BlockIndexX => "(blockIdx.x)";

            /// <summary>Yブロックインデクス</summary>
            public static string BlockIndexY => "(blockIdx.y)";

            /// <summary>Zブロックインデクス</summary>
            public static string BlockIndexZ => "(blockIdx.z)";

            /// <summary>XスレッドID</summary>
            public static string ThreadIdX => "(threadIdx.x)";

            /// <summary>YスレッドID</summary>
            public static string ThreadIdY => "(threadIdx.y)";

            /// <summary>Xスレッド数</summary>
            public static string ThreadsX => "(blockDim.x)";

            /// <summary>Yスレッド数</summary>
            public static string ThreadsY => "(blockDim.y)";
        }
    }
}
