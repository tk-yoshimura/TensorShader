namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>非数をゼロとして返す</summary>
    internal class NanAsZero : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public NanAsZero(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.NanAsZero((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
