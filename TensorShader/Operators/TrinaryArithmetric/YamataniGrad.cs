namespace TensorShader.Operators.TrinaryArithmetric {
    /// <summary>Yamatani勾配</summary>
    internal class YamataniGrad : TrinaryUniConstantArithmetric {
        /// <summary>コンストラクタ</summary>
        public YamataniGrad(float slope, Shape shape)
            : base(slope, shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Elementwise.YamataniGrad((uint)Shape.Length, Constant, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }
}
