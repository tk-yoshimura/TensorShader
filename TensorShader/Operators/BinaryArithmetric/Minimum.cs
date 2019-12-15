namespace TensorShader.Operators.BinaryArithmetric {
    /// <summary>最小値</summary>
    internal class Minimum : BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Minimum(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Elementwise.Minimum((uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }

    /// <summary>最小値</summary>
    internal class MinimumConstant : BinaryRightConstantArithmetric {
        /// <summary>コンストラクタ</summary>
        public MinimumConstant(float c, Shape shape)
            : base(c, shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.MinimumConstant((uint)Shape.Length, Constant, inmap.Buffer, outmap.Buffer);
        }
    }
}
