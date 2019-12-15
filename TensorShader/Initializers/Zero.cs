namespace TensorShader.Initializers {
    /// <summary>ゼロ初期化</summary>
    public class Zero : Initializer {
        /// <summary>コンストラクタ</summary>
        public Zero(Tensor tensor)
            : base(tensor) { }

        /// <summary>初期化</summary>
        public override void Execute() {
            Tensor.Zeroset();
        }
    }
}
