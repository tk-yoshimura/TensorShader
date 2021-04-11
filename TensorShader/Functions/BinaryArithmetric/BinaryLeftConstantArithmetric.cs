namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>2項演算</summary>
        internal static VariableNode BinaryLeftConstantArithmetric(VariableNode x, Operators.BinaryArithmetric.BinaryLeftConstantArithmetric binary_operator) {
            Function function = new Functions.BinaryLeftConstantArithmetric.BinaryLeftConstantArithmetric(binary_operator);

            return Apply(function, x)[0];
        }
    }

    public partial class Tensor {
        /// <summary>2項演算</summary>
        internal static Tensor BinaryLeftConstantArithmetric(Tensor x, Operators.BinaryArithmetric.BinaryLeftConstantArithmetric binary_operator) {
            Function function = new Functions.BinaryLeftConstantArithmetric.BinaryLeftConstantArithmetric(binary_operator);

            Shape y_shape = function.OutputShapes(x.Shape)[0];

            Tensor y = new(y_shape);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.BinaryLeftConstantArithmetric {
    /// <summary>2項演算</summary>
    internal class BinaryLeftConstantArithmetric : Function {
        private readonly Operator binary_operator;

        /// <summary>コンストラクタ</summary>
        public BinaryLeftConstantArithmetric(Operators.BinaryArithmetric.BinaryLeftConstantArithmetric binary_operator)
            : base(inputs: 1, outputs: 1, allow_resubstitution: true) {

            this.binary_operator = binary_operator;
        }

        public override string Name => binary_operator.Name;

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape shape = inshapes[0];

            return new Shape[] { shape };
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (new Tensor[] { intensor, outtensor }, binary_operator);
        }
    }
}
