namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>SoftPlus</summary>
    internal class SoftPlus : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public SoftPlus(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.SoftPlus((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
