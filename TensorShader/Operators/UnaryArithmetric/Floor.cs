namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>切り捨て関数</summary>
    internal class Floor : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Floor(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Floor((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
