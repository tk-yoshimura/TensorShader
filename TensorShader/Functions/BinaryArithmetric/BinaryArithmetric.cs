using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>2項演算</summary>
        internal static VariableNode BinaryArithmetric(VariableNode x1, VariableNode x2, Operators.BinaryArithmetric.BinaryArithmetric binary_operator) {
            Function function = new Functions.BinaryArithmetric.BinaryArithmetric(binary_operator);

            return Apply(function, x1, x2)[0];
        }
    }

    public partial class Tensor {
        /// <summary>2項演算</summary>
        internal static Tensor BinaryArithmetric(Tensor x1, Tensor x2, Operators.BinaryArithmetric.BinaryArithmetric binary_operator) {
            Function function = new Functions.BinaryArithmetric.BinaryArithmetric(binary_operator);

            Shape y_shape = function.OutputShapes(x1.Shape, x2.Shape)[0];

            Tensor y = new Tensor(y_shape);

            function.Execute(new Tensor[] { x1, x2 }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.BinaryArithmetric {
    /// <summary>2項演算</summary>
    internal class BinaryArithmetric : Function {
        private readonly Operator binary_operator;

        /// <summary>コンストラクタ</summary>
        public BinaryArithmetric(Operators.BinaryArithmetric.BinaryArithmetric binary_operator)
            : base(inputs: 2, outputs: 1, allow_resubstitution: true) {

            this.binary_operator = binary_operator;
        }

        public override string Name => binary_operator.Name;

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape shape = inshapes[0];

            return new Shape[] { shape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0] != inshapes[1]) {
                throw new ArgumentException(ExceptionMessage.Shape(inshapes[1], inshapes[0]));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor1 = intensors[0], intensor2 = intensors[1], outtensor = outtensors[0];

            return (new Tensor[] { intensor1, intensor2, outtensor }, binary_operator);
        }
    }
}
