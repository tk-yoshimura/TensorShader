using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>3項演算</summary>
        internal static VariableNode TrinaryArithmetric(VariableNode x1, VariableNode x2, VariableNode x3, Operators.TrinaryArithmetric.TrinaryArithmetric trinary_operator) {
            Function function = new Functions.TrinaryArithmetric.TrinaryArithmetric(trinary_operator);

            return Apply(function, x1, x2, x3)[0];
        }
    }

    public partial class Tensor {
        /// <summary>3項演算</summary>
        internal static Tensor TrinaryArithmetric(Tensor x1, Tensor x2, Tensor x3, Operators.TrinaryArithmetric.TrinaryArithmetric trinary_operator) {
            Function function = new Functions.TrinaryArithmetric.TrinaryArithmetric(trinary_operator);

            Shape y_shape = function.OutputShapes(x1.Shape, x2.Shape, x3.Shape)[0];

            Tensor y = new(y_shape);

            function.Execute(new Tensor[] { x1, x2, x3 }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.TrinaryArithmetric {
    /// <summary>3項演算</summary>
    internal class TrinaryArithmetric : Function {
        private readonly Operator trinary_operator;

        /// <summary>コンストラクタ</summary>
        public TrinaryArithmetric(Operators.TrinaryArithmetric.TrinaryArithmetric trinary_operator)
            : base(inputs: 3, outputs: 1, allow_resubstitution: true) {

            this.trinary_operator = trinary_operator;
        }

        public override string Name => trinary_operator.Name;

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape shape = inshapes[0];

            return new Shape[] { shape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0] != inshapes[2]) {
                throw new ArgumentException(ExceptionMessage.Shape(inshapes[0], inshapes[2]));
            }

            if (inshapes[1] != inshapes[2]) {
                throw new ArgumentException(ExceptionMessage.Shape(inshapes[1], inshapes[2]));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor1 = intensors[0], intensor2 = intensors[1], intensor3 = intensors[2], outtensor = outtensors[0];

            return (new Tensor[] { intensor1, intensor2, intensor3, outtensor }, trinary_operator);
        }
    }
}
