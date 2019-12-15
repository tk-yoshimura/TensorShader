namespace TensorShader.Operators.LogicalArithmetric {
    /// <summary>大なりイコール</summary>
    internal class GreaterThanOrEqualLeftConstant : BinaryArithmetric.BinaryLeftConstantArithmetric {
        /// <summary>コンストラクタ</summary>
        public GreaterThanOrEqualLeftConstant(float c, Shape shape)
            : base(c, shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.GreaterThanOrEqualConstant((uint)Shape.Length, Constant, inmap.Buffer, outmap.Buffer);
        }
    }
}
