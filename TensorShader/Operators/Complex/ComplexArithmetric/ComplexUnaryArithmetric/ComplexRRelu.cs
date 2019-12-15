namespace TensorShader.Operators.ComplexUnaryArithmetric {
    /// <summary>複素RRelu</summary>
    internal class ComplexRRelu : ComplexUnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexRRelu(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Complex.RRelu((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
