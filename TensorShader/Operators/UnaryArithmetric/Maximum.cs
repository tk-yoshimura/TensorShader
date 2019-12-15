namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>最大値</summary>
    internal class Maximum : UnaryArithmetric {
        /// <summary>定数項</summary>
        protected readonly float Constant;

        /// <summary>コンストラクタ</summary>
        public Maximum(float c, Shape shape)
            : base(shape) {
            this.Constant = c;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.MaximumConstant((uint)Shape.Length, Constant, inmap.Buffer, outmap.Buffer);
        }
    }
}
