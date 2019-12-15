namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>双曲線正弦関数</summary>
    internal class Sinh : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Sinh(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.Sinh((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
