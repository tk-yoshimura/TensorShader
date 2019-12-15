namespace TensorShader.Operators.BinaryArithmetric {
    /// <summary>除算</summary>
    internal class Div : BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Div(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Elementwise.Div((uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }

    /// <summary>除算</summary>
    internal class DivLeftVector : BinaryLeftVectorArithmetric {
        /// <summary>コンストラクタ</summary>
        public DivLeftVector(Shape vectorshape, Shape mapshape)
            : base(vectorshape, mapshape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Channelwise.DivLVector((uint)VectorShape.Length, (uint)MapShape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }

    /// <summary>除算</summary>
    internal class DivRightVector : BinaryRightVectorArithmetric {
        /// <summary>コンストラクタ</summary>
        public DivRightVector(Shape vectorshape, Shape mapshape)
            : base(vectorshape, mapshape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Channelwise.DivRVector((uint)VectorShape.Length, (uint)MapShape.Length, inmap2.Buffer, inmap1.Buffer, outmap.Buffer);
        }
    }

    /// <summary>除算</summary>
    internal class DivLeftConstant : BinaryLeftConstantArithmetric {
        /// <summary>コンストラクタ</summary>
        public DivLeftConstant(float c, Shape shape)
            : base(c, shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.DivConstant((uint)Shape.Length, Constant, inmap.Buffer, outmap.Buffer);
        }
    }

    /// <summary>除算</summary>
    internal class DivRightConstant : BinaryRightConstantArithmetric {
        /// <summary>コンストラクタ</summary>
        public DivRightConstant(float c, Shape shape)
            : base(c, shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.MulConstant((uint)Shape.Length, 1 / Constant, inmap.Buffer, outmap.Buffer);
        }
    }
}
