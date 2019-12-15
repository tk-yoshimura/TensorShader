namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>逆立方根</summary>
    internal class Rcbrt : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Rcbrt(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Rcbrt((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
