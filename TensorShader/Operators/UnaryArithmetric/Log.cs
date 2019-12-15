namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>対数関数</summary>
    internal class Log : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Log(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Log((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
