namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>2乗</summary>
    internal class Square : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Square(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Square((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
