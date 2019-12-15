namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>減算</summary>
    internal class Sub : UnaryArithmetric {
        /// <summary>定数項</summary>
        protected readonly float Constant;

        /// <summary>コンストラクタ</summary>
        public Sub(float c, Shape shape)
            : base(shape) {
            this.Constant = c;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.SubConstant((uint)Shape.Length, Constant, inmap.Buffer, outmap.Buffer);
        }
    }
}
