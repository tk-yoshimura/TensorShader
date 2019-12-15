using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>3次元ベクトルZ成分</summary>
        public static VariableNode TrivectorZ(VariableNode x) {
            Function function = new Functions.Trivector.TrivectorZ();

            return Apply(function, x)[0];
        }
    }

    public partial class Tensor {
        /// <summary>3次元ベクトルZ成分</summary>
        public static Tensor TrivectorZ(Tensor x) {
            Function function = new Functions.Trivector.TrivectorZ();

            Shape y_shape = function.OutputShapes(x.Shape)[0];

            Tensor y = new Tensor(y_shape);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Trivector {
    /// <summary>3次元ベクトルZ成分</summary>
    internal class TrivectorZ : Function {
        /// <summary>コンストラクタ</summary>
        public TrivectorZ()
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) { }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            int[] s = inshape;
            s[0] /= 3;

            return new Shape[] { new Shape(inshape.Type, s) };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Ndim <= 0) {
                throw new ArgumentException(ExceptionMessage.Shape("Ndim", inshapes[0]));
            }

            if (inshapes[0].Channels % 3 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("Channels", inshapes[0], inshapes[0].Channels, 3));
            }

            if (inshapes[0].InChannels % 3 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("InChannels", inshapes[0], inshapes[0].Channels, 3));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (new Tensor[] { intensor, outtensor },
                    new Operators.TrivectorCast.TrivectorZ(intensor.Shape));
        }
    }
}
