namespace TensorShader.Operators.TrinaryArithmetric {
    /// <summary>Clamp</summary>
    internal class Clamp : TrinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Clamp(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], inmap3 = tensors[2], outmap = tensors[3];

            TensorShaderCudaBackend.Elementwise.Clamp((uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, inmap3.Buffer, outmap.Buffer);
        }
    }

    /// <summary>Clamp</summary>
    internal class ClampConstant : TrinaryConstantArithmetric {
        /// <summary>コンストラクタ</summary>
        public ClampConstant(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], inmap3 = tensors[2], outmap = tensors[3];

            TensorShaderCudaBackend.Elementwise.ClampFactor((uint)Shape.Length, inmap2.Buffer, inmap3.Buffer, inmap1.Buffer, outmap.Buffer);
        }
    }
}
