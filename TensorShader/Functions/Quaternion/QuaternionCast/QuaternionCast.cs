using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>実数から四元数を構成</summary>
        public static VariableNode QuaternionCast(VariableNode real, VariableNode imag_i, VariableNode imag_j, VariableNode imag_k) {
            Function function = new Functions.Quaternion.QuaternionCast();

            return Apply(function, real, imag_i, imag_j, imag_k)[0];
        }
    }

    public partial class Tensor {
        /// <summary>実数から四元数を構成</summary>
        public static Tensor QuaternionCast(Tensor real, Tensor imag_i, Tensor imag_j, Tensor imag_k) {
            Function function = new Functions.Quaternion.QuaternionCast();

            Shape y_shape = function.OutputShapes(real.Shape)[0];

            Tensor y = new(y_shape);

            function.Execute(new Tensor[] { real, imag_i, imag_j, imag_k }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Quaternion {
    /// <summary>実数から複素数を構成</summary>
    internal class QuaternionCast : Function {
        /// <summary>コンストラクタ</summary>
        public QuaternionCast()
            : base(inputs: 4, outputs: 1, allow_resubstitution: false) { }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            int[] s = inshape;
            s[0] *= 4;

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
            if (inshapes[0] != inshapes[3]) {
                throw new ArgumentException(ExceptionMessage.Shape(inshapes[3], inshapes[0]));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor1 = intensors[0], intensor2 = intensors[1], intensor3 = intensors[2], intensor4 = intensors[3], outtensor = outtensors[0];

            return (new Tensor[] { intensor1, intensor2, intensor3, intensor4, outtensor },
                    new Operators.QuaternionCast.QuaternionCast(intensor1.Shape));
        }
    }
}
