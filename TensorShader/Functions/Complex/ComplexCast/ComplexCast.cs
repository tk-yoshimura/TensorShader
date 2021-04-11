using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>実数から複素数を構成</summary>
        public static VariableNode ComplexCast(VariableNode real, VariableNode imag) {
            Function function = new Functions.Complex.ComplexCast();

            return Apply(function, real, imag)[0];
        }
    }

    public partial class Tensor {
        /// <summary>実数から複素数を構成</summary>
        public static Tensor ComplexCast(Tensor real, Tensor imag) {
            Function function = new Functions.Complex.ComplexCast();

            Shape y_shape = function.OutputShapes(real.Shape)[0];

            Tensor y = new(y_shape);

            function.Execute(new Tensor[] { real, imag }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Complex {
    /// <summary>実数から複素数を構成</summary>
    internal class ComplexCast : Function {
        /// <summary>コンストラクタ</summary>
        public ComplexCast()
            : base(inputs: 2, outputs: 1, allow_resubstitution: false) { }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            int[] s = inshape;
            s[0] *= 2;

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
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor1 = intensors[0], intensor2 = intensors[1], outtensor = outtensors[0];

            return (new Tensor[] { intensor1, intensor2, outtensor },
                    new Operators.ComplexCast.ComplexCast(intensor1.Shape));
        }
    }
}
