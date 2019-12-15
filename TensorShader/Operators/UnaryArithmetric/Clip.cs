namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>範囲内に収める</summary>
    internal class Clip : UnaryArithmetric {
        private readonly float cmin, cmax;

        /// <summary>コンストラクタ</summary>
        public Clip(float cmin, float cmax, Shape shape)
            : base(shape) {
            this.cmin = cmin;
            this.cmax = cmax;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.ClampConstant((uint)Shape.Length, cmin, cmax, inmap.Buffer, outmap.Buffer);
        }
    }
}
