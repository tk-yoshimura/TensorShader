namespace TensorShader.Operators.TrivectorUnaryArithmetric {
    /// <summary>3次元ベクトルSquash</summary>
    internal class TrivectorSquash : TrivectorUnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public TrivectorSquash(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Trivector.Squash((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
