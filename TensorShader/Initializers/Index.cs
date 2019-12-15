namespace TensorShader.Initializers {
    /// <summary>インデクス</summary>
    public class Index : Initializer {
        private readonly Operators.ArrayManipulation.Index generator;

        /// <summary>コンストラクタ</summary>
        public Index(Tensor tensor, int axis)
            : base(tensor) {
            this.generator = new Operators.ArrayManipulation.Index(tensor.Shape, axis);
        }

        /// <summary>初期化フロー</summary>
        public override void Execute() {
            generator.Execute(Tensor);
        }
    }
}
