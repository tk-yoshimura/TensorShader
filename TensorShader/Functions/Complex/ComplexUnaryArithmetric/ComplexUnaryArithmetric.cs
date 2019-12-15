using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素数1項演算</summary>
        internal static VariableNode ComplexUnaryArithmetric(VariableNode x, Operators.ComplexUnaryArithmetric.ComplexUnaryArithmetric arithmetric) {
            Function function = new Functions.ComplexArithmetric.ComplexUnaryArithmetric(arithmetric);

            return Apply(function, x)[0];
        }
    }

    public partial class Tensor {
        /// <summary>複素数1項演算</summary>
        internal static Tensor ComplexUnaryArithmetric(Tensor x, Operators.ComplexUnaryArithmetric.ComplexUnaryArithmetric arithmetric) {
            Function function = new Functions.ComplexArithmetric.ComplexUnaryArithmetric(arithmetric);

            Shape y_shape = function.OutputShapes(x.Shape)[0];

            Tensor y = new Tensor(y_shape);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.ComplexArithmetric {
    /// <summary>複素数1項演算</summary>
    internal class ComplexUnaryArithmetric : Function {
        private readonly Operator arithmetric;

        /// <summary>コンストラクタ</summary>
        public ComplexUnaryArithmetric(Operators.ComplexUnaryArithmetric.ComplexUnaryArithmetric arithmetric)
            : base(inputs: 1, outputs: 1, allow_resubstitution: true) {
            this.arithmetric = arithmetric;
        }

        public override string Name => arithmetric.Name;

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return inshapes;
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Ndim <= 0) {
                throw new ArgumentException(ExceptionMessage.Shape("Ndim", inshapes[0]));
            }

            if (inshapes[0].Channels % 2 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("Channels", inshapes[0], inshapes[0].Channels, 2));
            }

            if (inshapes[0].InChannels % 2 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("InChannels", inshapes[0], inshapes[0].Channels, 2));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (new Tensor[] { intensor, outtensor }, arithmetric);
        }
    }
}
