namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>正弦関数</summary>
    internal class Sin : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Sin(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Sin((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
