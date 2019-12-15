namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>平方根</summary>
    internal class Sqrt : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Sqrt(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Sqrt((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
