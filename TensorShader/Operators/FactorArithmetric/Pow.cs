namespace TensorShader.Operators.FactorArithmetric {
    /// <summary>べき乗</summary>
    internal class Pow : FactorArithmetric {
        /// <summary>コンストラクタ</summary>
        public Pow(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Elementwise.PowFactor((uint)Shape.Length, inmap2.Buffer, inmap1.Buffer, outmap.Buffer);
        }
    }
}
