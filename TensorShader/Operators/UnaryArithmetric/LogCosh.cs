namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>対数双曲線余弦関数</summary>
    internal class LogCosh : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public LogCosh(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.LogCosh((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
