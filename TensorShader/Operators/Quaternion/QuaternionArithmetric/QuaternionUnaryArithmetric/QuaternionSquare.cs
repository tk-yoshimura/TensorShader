namespace TensorShader.Operators.QuaternionUnaryArithmetric {
    /// <summary>四元数2乗</summary>
    internal class QuaternionSquare : QuaternionUnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public QuaternionSquare(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Quaternion.Square((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
