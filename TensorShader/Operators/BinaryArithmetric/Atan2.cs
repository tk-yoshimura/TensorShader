namespace TensorShader.Operators.BinaryArithmetric {
    /// <summary>全象限対応逆正接関数</summary>
    internal class Arctan2 : BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Arctan2(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Elementwise.ArcTan2((uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }
}
