namespace TensorShader.Operators.LogicalArithmetric {
    /// <summary>論理否定</summary>
    internal class LogicalNot : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public LogicalNot(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.LogicalNot((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
