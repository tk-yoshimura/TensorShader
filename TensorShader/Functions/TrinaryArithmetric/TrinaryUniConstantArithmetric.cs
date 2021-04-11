using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>3項演算</summary>
        internal static VariableNode TrinaryUniConstantArithmetric(VariableNode x1, VariableNode x2, Operators.TrinaryArithmetric.TrinaryUniConstantArithmetric trinary_operator) {
            Function function = new Functions.TrinaryArithmetric.TrinaryUniConstantArithmetric(trinary_operator);

            return Apply(function, x1, x2)[0];
        }
    }

    public partial class Tensor {
        /// <summary>3項演算</summary>
        internal static Tensor TrinaryUniConstantArithmetric(Tensor x1, Tensor x2, Operators.TrinaryArithmetric.TrinaryUniConstantArithmetric trinary_operator) {
            Function function = new Functions.TrinaryArithmetric.TrinaryUniConstantArithmetric(trinary_operator);

            Shape y_shape = function.OutputShapes(x1.Shape, x2.Shape)[0];

            Tensor y = new(y_shape);

            function.Execute(new Tensor[] { x1, x2 }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.TrinaryArithmetric {
    /// <summary>3項演算</summary>
    internal class TrinaryUniConstantArithmetric : Function {
        private readonly Operator trinary_operator;

        /// <summary>コンストラクタ</summary>
        public TrinaryUniConstantArithmetric(Operators.TrinaryArithmetric.TrinaryUniConstantArithmetric trinary_operator)
            : base(inputs: 2, outputs: 1, allow_resubstitution: true) {

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

            if (inshapes[0] != inshapes[1]) {
                throw new ArgumentException(ExceptionMessage.Shape(inshapes[0], inshapes[1]));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor1 = intensors[0], intensor2 = intensors[1], outtensor = outtensors[0];

            return (new Tensor[] { intensor1, intensor2, outtensor }, trinary_operator);
        }
    }
}
