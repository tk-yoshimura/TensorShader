namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>繰り上げ関数</summary>
    internal class Ceil : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Ceil(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Ceil((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
