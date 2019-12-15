namespace TensorShader.Operators.BinaryArithmetric {
    /// <summary>加算</summary>
    internal class Add : BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Add(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Elementwise.Add((uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }

    /// <summary>加算</summary>
    internal class AddLeftVector : BinaryLeftVectorArithmetric {
        /// <summary>コンストラクタ</summary>
        public AddLeftVector(Shape vectorshape, Shape mapshape)
            : base(vectorshape, mapshape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Channelwise.Add((uint)VectorShape.Length, (uint)MapShape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }

    /// <summary>加算</summary>
    internal class AddRightVector : BinaryRightVectorArithmetric {
        /// <summary>コンストラクタ</summary>
        public AddRightVector(Shape vectorshape, Shape mapshape)
            : base(vectorshape, mapshape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Channelwise.Add((uint)VectorShape.Length, (uint)MapShape.Length, inmap2.Buffer, inmap1.Buffer, outmap.Buffer);
        }
    }

    /// <summary>加算</summary>
    internal class AddLeftConstant : BinaryLeftConstantArithmetric {
        /// <summary>コンストラクタ</summary>
        public AddLeftConstant(float c, Shape shape)
            : base(c, shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.AddConstant((uint)Shape.Length, Constant, inmap.Buffer, outmap.Buffer);
        }
    }

    /// <summary>加算</summary>
    internal class AddRightConstant : BinaryRightConstantArithmetric {
        /// <summary>コンストラクタ</summary>
        public AddRightConstant(float c, Shape shape)
            : base(c, shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.AddConstant((uint)Shape.Length, Constant, inmap.Buffer, outmap.Buffer);
        }
    }
}
