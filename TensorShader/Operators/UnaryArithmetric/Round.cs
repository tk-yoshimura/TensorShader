namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>最近傍整数丸め関数</summary>
    internal class Round : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Round(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Round((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
