namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>カスタム</summary>
    internal class Custom : UnaryArithmetric {
        private readonly string funcname, funccode;

        /// <summary>コンストラクタ</summary>
        public Custom(Shape shape, string funcname, string funccode)
            : base(shape) {

            this.funcname = funcname;
            this.funccode = funccode;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.UnaryArithmetric((uint)Shape.Length, inmap.Buffer, outmap.Buffer, funcname, funccode);
        }
    }
}
