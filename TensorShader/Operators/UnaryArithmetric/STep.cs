namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>単位ステップ関数</summary>
    internal class Step : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Step(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Step((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
