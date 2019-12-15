namespace TensorShader.Operators.Aggregation {
    /// <summary>最大値</summary>
    internal class Max : Aggregation {
        /// <summary>コンストラクタ</summary>
        public Max(Shape shape, int axis)
            : base(shape, axis) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Aggregation.Max(AxisLength * Stride, inmap.Buffer, Stride, outmap.Buffer, Slides);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
