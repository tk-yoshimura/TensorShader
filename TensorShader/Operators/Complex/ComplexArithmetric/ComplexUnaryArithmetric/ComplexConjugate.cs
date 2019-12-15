namespace TensorShader.Operators.ComplexUnaryArithmetric {
    /// <summary>複素共役</summary>
    internal class ComplexConjugate : ComplexUnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexConjugate(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Complex.Conjugate((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
