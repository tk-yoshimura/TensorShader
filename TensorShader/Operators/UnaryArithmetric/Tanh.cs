namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>双曲線正接関数</summary>
    internal class Tanh : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Tanh(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Tanh((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
