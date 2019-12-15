namespace TensorShader.Operators.BinaryArithmetric {
    /// <summary>Relu勾配</summary>
    internal class ReluGrad : BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ReluGrad(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Elementwise.ReluGrad((uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }
}
