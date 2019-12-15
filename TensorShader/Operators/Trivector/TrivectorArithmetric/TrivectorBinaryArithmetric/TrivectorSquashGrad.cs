namespace TensorShader.Operators.TrivectorBinaryArithmetric {
    /// <summary>3次元ベクトルSquash勾配</summary>
    internal class TrivectorSquashGrad : TrivectorBinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public TrivectorSquashGrad(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Trivector.SquashGrad((uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }
}
