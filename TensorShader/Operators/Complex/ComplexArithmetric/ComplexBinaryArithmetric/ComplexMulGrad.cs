namespace TensorShader.Operators.ComplexBinaryArithmetric {
    /// <summary>複素乗算勾配</summary>
    internal class ComplexMulGrad : ComplexBinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexMulGrad(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Complex.MulGrad((uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }
}
