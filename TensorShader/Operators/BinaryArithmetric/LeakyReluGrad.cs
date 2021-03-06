namespace TensorShader.Operators.BinaryArithmetric {
    /// <summary>LeakyRelu勾配</summary>
    internal class LeakyReluGrad : BinaryArithmetric {
        protected readonly float Slope;

        /// <summary>コンストラクタ</summary>
        public LeakyReluGrad(float slope, Shape shape)
            : base(shape) {
            this.Slope = slope;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Elementwise.LeakyReluGrad((uint)Shape.Length, Slope, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }
}
