namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>1項演算</summary>
        internal static VariableNode UnaryArithmetric(VariableNode x, Operators.UnaryArithmetric.UnaryArithmetric unary_operator) {
            Function function = new Functions.UnaryArithmetric.UnaryArithmetric(unary_operator);

            return Apply(function, x)[0];
        }
    }

    public partial class Tensor {
        /// <summary>1項演算</summary>
        internal static Tensor UnaryArithmetric(Tensor x, Operators.UnaryArithmetric.UnaryArithmetric unary_operator) {
            Function function = new Functions.UnaryArithmetric.UnaryArithmetric(unary_operator);

            Shape y_shape = function.OutputShapes(x.Shape)[0];

            Tensor y = new Tensor(y_shape);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.UnaryArithmetric {
    /// <summary>1項演算</summary>
    internal class UnaryArithmetric : Function {
        private readonly Operator unary_operator;

        /// <summary>コンストラクタ</summary>
        public UnaryArithmetric(Operators.UnaryArithmetric.UnaryArithmetric unary_operator)
             : base(inputs: 1, outputs: 1, allow_resubstitution: true) {
            this.unary_operator = unary_operator;
        }

        public override string Name => unary_operator.Name;

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return inshapes;
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (new Tensor[] { intensor, outtensor }, unary_operator);
        }
    }
}
