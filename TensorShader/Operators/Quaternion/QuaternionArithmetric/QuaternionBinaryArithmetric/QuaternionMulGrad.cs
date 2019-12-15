namespace TensorShader.Operators.QuaternionBinaryArithmetric {
    /// <summary>四元数積勾配</summary>
    internal class QuaternionMulGrad : QuaternionBinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public QuaternionMulGrad(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Quaternion.MulGrad((uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }
}
