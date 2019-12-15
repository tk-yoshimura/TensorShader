namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>2進対数関数</summary>
    internal class Log2 : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Log2(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Log2((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
