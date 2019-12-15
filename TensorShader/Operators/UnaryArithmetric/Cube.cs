namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>3乗</summary>
    internal class Cube : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Cube(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Cube((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
