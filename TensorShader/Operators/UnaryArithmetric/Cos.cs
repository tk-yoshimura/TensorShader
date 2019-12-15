namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>余弦関数</summary>
    internal class Cos : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Cos(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Cos((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
