namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>符号関数</summary>
    internal class Sign : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Sign(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Sign((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
