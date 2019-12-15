namespace TensorShader.Operators.TrivectorUnaryArithmetric {
    /// <summary>3次元ベクトルDecay</summary>
    internal class TrivectorDecay : TrivectorUnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public TrivectorDecay(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Trivector.Decay((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
