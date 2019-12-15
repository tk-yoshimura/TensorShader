namespace TensorShader.Operators.BinaryArithmetric {
    /// <summary>乗算</summary>
    internal class Mul : BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Mul(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Elementwise.Mul((uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }

    /// <summary>乗算</summary>
    internal class MulLeftVector : BinaryLeftVectorArithmetric {
        /// <summary>コンストラクタ</summary>
        public MulLeftVector(Shape vectorshape, Shape mapshape)
            : base(vectorshape, mapshape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Channelwise.Mul((uint)VectorShape.Length, (uint)MapShape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }

    /// <summary>乗算</summary>
    internal class MulRightVector : BinaryRightVectorArithmetric {
        /// <summary>コンストラクタ</summary>
        public MulRightVector(Shape vectorshape, Shape mapshape)
            : base(vectorshape, mapshape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Channelwise.Mul((uint)VectorShape.Length, (uint)MapShape.Length, inmap2.Buffer, inmap1.Buffer, outmap.Buffer);
        }
    }

    /// <summary>乗算</summary>
    internal class MulLeftConstant : BinaryLeftConstantArithmetric {
        /// <summary>コンストラクタ</summary>
        public MulLeftConstant(float c, Shape shape)
            : base(c, shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.MulConstant((uint)Shape.Length, Constant, inmap.Buffer, outmap.Buffer);
        }
    }

    /// <summary>乗算</summary>
    internal class MulRightConstant : BinaryRightConstantArithmetric {
        /// <summary>コンストラクタ</summary>
        public MulRightConstant(float c, Shape shape)
            : base(c, shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.MulConstant((uint)Shape.Length, Constant, inmap.Buffer, outmap.Buffer);
        }
    }
}
