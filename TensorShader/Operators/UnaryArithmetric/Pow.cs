namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>べき乗</summary>
    internal class Pow : UnaryArithmetric {
        /// <summary>定数項</summary>
        protected readonly float Constant;

        /// <summary>コンストラクタ</summary>
        public Pow(float c, Shape shape)
            : base(shape) {
            this.Constant = c;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.PowConstant((uint)Shape.Length, Constant, inmap.Buffer, outmap.Buffer);
        }
    }
}
