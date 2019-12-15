namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>逆正弦関数</summary>
    internal class Arcsin : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Arcsin(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.ArcSin((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
