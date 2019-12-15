namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>逆余弦関数</summary>
    internal class Arccos : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Arccos(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.ArcCos((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
