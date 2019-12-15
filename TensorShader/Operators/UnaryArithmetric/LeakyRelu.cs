namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>LeakyRelu</summary>
    internal class LeakyRelu : UnaryArithmetric {
        private readonly float slope;

        /// <summary>コンストラクタ</summary>
        public LeakyRelu(float slope, Shape shape)
            : base(shape) {
            this.slope = slope;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.LeakyRelu((uint)Shape.Length, slope, inmap.Buffer, outmap.Buffer);
        }
    }
}
