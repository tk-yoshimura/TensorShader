using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数2項演算</summary>
        internal static VariableNode QuaternionBinaryArithmetric(VariableNode x1, VariableNode x2, Operators.QuaternionBinaryArithmetric.QuaternionBinaryArithmetric arithmetric) {
            Function function = new Functions.QuaternionArithmetric.QuaternionBinaryArithmetric(arithmetric);

            return Apply(function, x1, x2)[0];
        }
    }

    public partial class Tensor {
        /// <summary>四元数2項演算</summary>
        internal static Tensor QuaternionBinaryArithmetric(Tensor x1, Tensor x2, Operators.QuaternionBinaryArithmetric.QuaternionBinaryArithmetric arithmetric) {
            Function function = new Functions.QuaternionArithmetric.QuaternionBinaryArithmetric(arithmetric);

            Shape y_shape = function.OutputShapes(x1.Shape, x2.Shape)[0];

            Tensor y = new Tensor(y_shape);

            function.Execute(new Tensor[] { x1, x2 }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.QuaternionArithmetric {
    /// <summary>四元数2項演算</summary>
    internal class QuaternionBinaryArithmetric : Function {
        private readonly Operator arithmetric;

        /// <summary>コンストラクタ</summary>
        public QuaternionBinaryArithmetric(Operators.QuaternionBinaryArithmetric.QuaternionBinaryArithmetric arithmetric)
            : base(inputs: 2, outputs: 1, allow_resubstitution: true) {
            this.arithmetric = arithmetric;
        }

        public override string Name => arithmetric.Name;

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return new Shape[] { inshapes[0] };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Ndim <= 0) {
                throw new ArgumentException(ExceptionMessage.Shape("Ndim", inshapes[0]));
            }

            if (inshapes[0].Channels % 4 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("Channels", inshapes[0], inshapes[0].Channels, 4));
            }

            if (inshapes[0].InChannels % 4 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("InChannels", inshapes[0], inshapes[0].Channels, 4));
            }

            if (inshapes[0] != inshapes[1]) {
                throw new ArgumentException(ExceptionMessage.Shape(inshapes[1], inshapes[0]));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor1 = intensors[0], intensor2 = intensors[1], outtensor = outtensors[0];

            return (new Tensor[] { intensor1, intensor2, outtensor }, arithmetric);
        }
    }
}
