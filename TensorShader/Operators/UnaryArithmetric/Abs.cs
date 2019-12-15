namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>絶対値</summary>
    internal class Abs : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Abs(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Abs((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
