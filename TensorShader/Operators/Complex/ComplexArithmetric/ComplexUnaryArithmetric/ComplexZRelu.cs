namespace TensorShader.Operators.ComplexUnaryArithmetric {
    /// <summary>複素ZRelu</summary>
    internal class ComplexZRelu : ComplexUnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexZRelu(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Complex.ZRelu((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
