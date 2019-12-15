using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>実数から3次元ベクトルを構成</summary>
        public static VariableNode TrivectorCast(VariableNode x, VariableNode y, VariableNode z) {
            Function function = new Functions.Trivector.TrivectorCast();

            return Apply(function, x, y, z)[0];
        }
    }

    public partial class Tensor {
        /// <summary>実数から3次元ベクトルを構成</summary>
        public static Tensor TrivectorCast(Tensor x, Tensor y, Tensor z) {
            Function function = new Functions.Trivector.TrivectorCast();

            Shape v_shape = function.OutputShapes(x.Shape)[0];

            Tensor v = new Tensor(v_shape);

            function.Execute(new Tensor[] { x, y, z }, new Tensor[] { v });

            return y;
        }
    }
}

namespace TensorShader.Functions.Trivector {
    /// <summary>実数から複素数を構成</summary>
    internal class TrivectorCast : Function {
        /// <summary>コンストラクタ</summary>
        public TrivectorCast()
            : base(inputs: 3, outputs: 1, allow_resubstitution: false) { }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            int[] s = inshape;
            s[0] *= 3;

            return new Shape[] { new Shape(inshape.Type, s) };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Ndim <= 0) {
                throw new ArgumentException(ExceptionMessage.Shape("Ndim", inshapes[0]));
            }

            if (inshapes[0] != inshapes[1]) {
                throw new ArgumentException(ExceptionMessage.Shape(inshapes[1], inshapes[0]));
            }
            if (inshapes[0] != inshapes[2]) {
                throw new ArgumentException(ExceptionMessage.Shape(inshapes[2], inshapes[0]));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor1 = intensors[0], intensor2 = intensors[1], intensor3 = intensors[2], outtensor = outtensors[0];

            return (new Tensor[] { intensor1, intensor2, intensor3, outtensor },
                    new Operators.TrivectorCast.TrivectorCast(intensor1.Shape));
        }
    }
}
