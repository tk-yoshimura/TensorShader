namespace TensorShader.Operators.QuaternionUnaryArithmetric {
    /// <summary>四元数RRelu</summary>
    internal class QuaternionRRelu : QuaternionUnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public QuaternionRRelu(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Quaternion.RRelu((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
