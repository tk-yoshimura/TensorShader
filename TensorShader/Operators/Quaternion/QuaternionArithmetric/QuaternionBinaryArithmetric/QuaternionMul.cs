namespace TensorShader.Operators.QuaternionBinaryArithmetric {
    /// <summary>四元数積</summary>
    internal class QuaternionMul : QuaternionBinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public QuaternionMul(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Quaternion.Mul((uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }
}
