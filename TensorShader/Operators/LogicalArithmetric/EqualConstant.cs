namespace TensorShader.Operators.LogicalArithmetric {
    /// <summary>イコール</summary>
    internal class EqualConstant : BinaryArithmetric.BinaryLeftConstantArithmetric {
        /// <summary>コンストラクタ</summary>
        public EqualConstant(float c, Shape shape)
            : base(c, shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.EqualConstant((uint)Shape.Length, Constant, inmap.Buffer, outmap.Buffer);
        }
    }
}
