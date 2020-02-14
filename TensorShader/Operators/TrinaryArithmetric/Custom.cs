namespace TensorShader.Operators.TrinaryArithmetric {
    /// <summary>カスタム</summary>
    internal class CustomTrinaryArithmetric : TrinaryArithmetric {
        private readonly string funcname, funccode;

        /// <summary>コンストラクタ</summary>
        public CustomTrinaryArithmetric(Shape shape, string funcname, string funccode)
            : base(shape) {

            this.funcname = funcname;
            this.funccode = funccode;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], inmap3 = tensors[2], outmap = tensors[3];

            TensorShaderCudaBackend.Elementwise.TrinaryArithmetric((uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, inmap3.Buffer, outmap.Buffer, funcname, funccode);
        }
    }

    /// <summary>カスタム</summary>
    internal class CustomTrinaryUniConstantArithmetric : TrinaryUniConstantArithmetric {
        private readonly string funcname, funccode;

        /// <summary>コンストラクタ</summary>
        public CustomTrinaryUniConstantArithmetric(float c, Shape shape, string funcname, string funccode)
            : base(c, shape) { 
        
            this.funcname = funcname;
            this.funccode = funccode;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderCudaBackend.Elementwise.TrinaryUniConstantArithmetric((uint)Shape.Length, Constant, inmap1.Buffer, inmap2.Buffer, outmap.Buffer, funcname, funccode);
        }
    }

    /// <summary>カスタム</summary>
    internal class CustomTrinaryBiConstantArithmetric : TrinaryBiConstantArithmetric {
        private readonly string funcname, funccode;

        /// <summary>コンストラクタ</summary>
        public CustomTrinaryBiConstantArithmetric(float c1, float c2, Shape shape, string funcname, string funccode)
            : base(c1, c2, shape) { 
        
            this.funcname = funcname;
            this.funccode = funccode;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Elementwise.TrinaryBiConstantArithmetric((uint)Shape.Length, Constant1, Constant2, inmap.Buffer, outmap.Buffer, funcname, funccode);
        }
    }
}
