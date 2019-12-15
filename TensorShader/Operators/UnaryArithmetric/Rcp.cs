namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>逆数関数</summary>
    internal class Rcp : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Rcp(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Rcp((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
