namespace TensorShader.Operators.TrinaryArithmetric {
    /// <summary>Yamatani</summary>
    internal class Yamatani : TrinaryUniConstantArithmetric {
        /// <summary>コンストラクタ</summary>
        public Yamatani(float slope, Shape shape)
            : base(slope, shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Elementwise.Yamatani((uint)Shape.Length, Constant, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }
}
