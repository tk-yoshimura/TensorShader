namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>正接関数</summary>
    internal class Tan : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Tan(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Tan((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
