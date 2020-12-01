using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>3次元ベクトル2項演算</summary>
        internal static VariableNode TrivectorBinaryArithmetric(VariableNode v, VariableNode u, Operators.TrivectorBinaryArithmetric.TrivectorBinaryArithmetric arithmetric) {
            Function function = new Functions.TrivectorArithmetric.TrivectorBinaryArithmetric(arithmetric);

            return Apply(function, v, u)[0];
        }
    }

    public partial class Tensor {
        /// <summary>3次元ベクトル2項演算</summary>
        internal static Tensor TrivectorBinaryArithmetric(Tensor v, Tensor u, Operators.TrivectorBinaryArithmetric.TrivectorBinaryArithmetric arithmetric) {
            Function function = new Functions.TrivectorArithmetric.TrivectorBinaryArithmetric(arithmetric);

            Shape y_shape = function.OutputShapes(v.Shape, u.Shape)[0];

            Tensor y = new Tensor(y_shape);

            function.Execute(new Tensor[] { v, u }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.TrivectorArithmetric {
    /// <summary>3次元ベクトル2項演算</summary>
    internal class TrivectorBinaryArithmetric : Function {
        private readonly Operator arithmetric;

        /// <summary>コンストラクタ</summary>
        public TrivectorBinaryArithmetric(Operators.TrivectorBinaryArithmetric.TrivectorBinaryArithmetric arithmetric)
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

            if (inshapes[0].Channels % 3 != 0) {
                throw new AggregateException(ExceptionMessage.LengthMultiple("Channels", inshapes[0], inshapes[0].Channels, 3));
            }

            if (inshapes[0].InChannels % 3 != 0) {
                throw new AggregateException(ExceptionMessage.LengthMultiple("InChannels", inshapes[0], inshapes[0].Channels, 3));
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
