namespace TensorShader.Operators.ComplexUnaryArithmetric {
    /// <summary>複素Squash</summary>
    internal class ComplexSquash : ComplexUnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexSquash(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Complex.Squash((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
