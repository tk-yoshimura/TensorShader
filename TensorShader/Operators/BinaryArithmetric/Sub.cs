namespace TensorShader.Operators.BinaryArithmetric {
    /// <summary>減算</summary>
    internal class Sub : BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Sub(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Elementwise.Sub((uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }

    /// <summary>減算</summary>
    internal class SubLeftVector : BinaryLeftVectorArithmetric {
        /// <summary>コンストラクタ</summary>
        public SubLeftVector(Shape vectorshape, Shape mapshape)
            : base(vectorshape, mapshape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Channelwise.SubLVector((uint)VectorShape.Length, (uint)MapShape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }

    /// <summary>減算</summary>
    internal class SubRightVector : BinaryRightVectorArithmetric {
        /// <summary>コンストラクタ</summary>
        public SubRightVector(Shape vectorshape, Shape mapshape)
            : base(vectorshape, mapshape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Channelwise.SubRVector((uint)VectorShape.Length, (uint)MapShape.Length, inmap2.Buffer, inmap1.Buffer, outmap.Buffer);
        }
    }

    /// <summary>減算</summary>
    internal class SubLeftConstant : BinaryLeftConstantArithmetric {
        /// <summary>コンストラクタ</summary>
        public SubLeftConstant(float c, Shape shape)
            : base(c, shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.SubConstant((uint)Shape.Length, Constant, inmap.Buffer, outmap.Buffer);
        }
    }

    /// <summary>減算</summary>
    internal class SubRightConstant : BinaryRightConstantArithmetric {
        /// <summary>コンストラクタ</summary>
        public SubRightConstant(float c, Shape shape)
            : base(c, shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.AddConstant((uint)Shape.Length, -Constant, inmap.Buffer, outmap.Buffer);
        }
    }
}
