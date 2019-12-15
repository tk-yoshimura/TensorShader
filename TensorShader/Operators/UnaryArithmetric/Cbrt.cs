namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>立方根</summary>
    internal class Cbrt : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Cbrt(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Cbrt((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
