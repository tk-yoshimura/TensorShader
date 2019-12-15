namespace TensorShader.Initializers {
    /// <summary>定数初期化</summary>
    public class Constant : Initializer {
        /// <summary>定数</summary>
        public float Val { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Constant(Tensor tensor, float val)
            : base(tensor) {
            this.Val = val;
        }

        /// <summary>初期化フロー</summary>
        public override void Execute() {
            Tensor.Clear(Val);
        }
    }
}
