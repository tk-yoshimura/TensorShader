namespace TensorShader.Operators.QuaternionUnaryArithmetric {
    /// <summary>四元数正規化</summary>
    internal class QuaternionNormalize : QuaternionUnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public QuaternionNormalize(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Quaternion.Normalize((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
