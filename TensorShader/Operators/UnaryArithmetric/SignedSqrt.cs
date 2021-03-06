namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>符号付き平方根</summary>
    internal class SignedSqrt : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public SignedSqrt(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.SignedSqrt((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
