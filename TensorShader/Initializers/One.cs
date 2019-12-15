namespace TensorShader.Initializers {
    /// <summary>1初期化</summary>
    public class One : Constant {
        /// <summary>コンストラクタ</summary>
        public One(Tensor tensor)
            : base(tensor, 1) { }
    }
}
