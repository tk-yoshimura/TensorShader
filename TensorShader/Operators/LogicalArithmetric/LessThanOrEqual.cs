namespace TensorShader.Operators.LogicalArithmetric {
    /// <summary>小なりイコール</summary>
    internal class LessThanOrEqual : BinaryArithmetric.BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public LessThanOrEqual(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Elementwise.LessThanOrEqual((uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }
}
