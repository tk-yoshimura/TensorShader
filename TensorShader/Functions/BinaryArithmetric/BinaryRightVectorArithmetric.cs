using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>2項演算</summary>
        internal static VariableNode BinaryRightVectorArithmetric(VariableNode x1, VariableNode x2, Operators.BinaryArithmetric.BinaryRightVectorArithmetric binary_operator) {
            Function function = new Functions.BinaryRightVectorArithmetric.BinaryRightVectorArithmetric(binary_operator);

            return Apply(function, x1, x2)[0];
        }
    }

    public partial class Tensor {
        /// <summary>2項演算</summary>
        internal static Tensor BinaryRightVectorArithmetric(Tensor x1, Tensor x2, Operators.BinaryArithmetric.BinaryRightVectorArithmetric binary_operator) {
            Function function = new Functions.BinaryRightVectorArithmetric.BinaryRightVectorArithmetric(binary_operator);

            Shape y_shape = function.OutputShapes(x1.Shape, x2.Shape)[0];

            Tensor y = new Tensor(y_shape);

            function.Execute(new Tensor[] { x1, x2 }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.BinaryRightVectorArithmetric {
    /// <summary>2項演算</summary>
    internal class BinaryRightVectorArithmetric : Function {
        private readonly Operator binary_operator;

        /// <summary>コンストラクタ</summary>
        public BinaryRightVectorArithmetric(Operators.BinaryArithmetric.BinaryRightVectorArithmetric binary_operator)
            : base(inputs: 2, outputs: 1, allow_resubstitution: false) {
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

            if (inshapes[0].Ndim <= inshapes[1].Ndim) {
                throw new ArgumentException(ExceptionMessage.Vectorize(inshapes[0], inshapes[1]));
            }

            for (int i = 0; i < inshapes[1].Ndim; i++) {
                if (inshapes[0][i] != inshapes[1][i]) {
                    throw new ArgumentException(ExceptionMessage.Vectorize(inshapes[0], inshapes[1]));
                }
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
