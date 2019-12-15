namespace TensorShader.Operators.ComplexUnaryArithmetric {
    /// <summary>複素正規化</summary>
    internal class ComplexNormalize : ComplexUnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexNormalize(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Complex.Normalize((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
