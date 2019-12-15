namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>2項演算</summary>
        internal static VariableNode BinaryRightConstantArithmetric(VariableNode x, Operators.BinaryArithmetric.BinaryRightConstantArithmetric binary_operator) {
            Function function = new Functions.BinaryRightConstantArithmetric.BinaryRightConstantArithmetric(binary_operator);

            return Apply(function, x)[0];
        }
    }

    public partial class Tensor {
        /// <summary>2項演算</summary>
        internal static Tensor BinaryRightConstantArithmetric(Tensor x, Operators.BinaryArithmetric.BinaryRightConstantArithmetric binary_operator) {
            Function function = new Functions.BinaryRightConstantArithmetric.BinaryRightConstantArithmetric(binary_operator);

            Shape y_shape = function.OutputShapes(x.Shape)[0];

            Tensor y = new Tensor(y_shape);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.BinaryRightConstantArithmetric {
    /// <summary>2項演算</summary>
    internal class BinaryRightConstantArithmetric : Function {
        private readonly Operator binary_operator;

        /// <summary>コンストラクタ</summary>
        public BinaryRightConstantArithmetric(Operators.BinaryArithmetric.BinaryRightConstantArithmetric binary_operator)
            : base(inputs: 1, outputs: 1, allow_resubstitution: true) {
            this.binary_operator = binary_operator;
        }

        public override string Name => binary_operator.Name;

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return inshapes;
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (new Tensor[] { intensor, outtensor }, binary_operator);
        }
    }
}
