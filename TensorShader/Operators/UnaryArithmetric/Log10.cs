namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>自然対数関数</summary>
    internal class Log10 : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Log10(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Log10((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
