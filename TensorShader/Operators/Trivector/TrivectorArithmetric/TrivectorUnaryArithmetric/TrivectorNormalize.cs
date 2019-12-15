namespace TensorShader.Operators.TrivectorUnaryArithmetric {
    /// <summary>3次元ベクトル正規化</summary>
    internal class TrivectorNormalize : TrivectorUnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public TrivectorNormalize(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Trivector.Normalize((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
