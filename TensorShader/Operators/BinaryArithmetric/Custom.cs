namespace TensorShader.Operators.BinaryArithmetric {
    /// <summary>カスタム</summary>
    internal class Custom : BinaryArithmetric {
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

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Elementwise.BinaryArithmetric((uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer, funcname, funccode);
        }
    }
}
