namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>Elu</summary>
    internal class Elu : UnaryArithmetric {
        private readonly float slope;

        /// <summary>コンストラクタ</summary>
        public Elu(float slope, Shape shape)
            : base(shape) {
            this.slope = slope;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Elu((uint)Shape.Length, slope, inmap.Buffer, outmap.Buffer);
        }
    }
}
