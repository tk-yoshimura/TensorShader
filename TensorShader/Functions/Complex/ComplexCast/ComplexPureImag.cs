using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素純虚数</summary>
        public static VariableNode ComplexPureImag(VariableNode real) {
            Function function = new Functions.Complex.ComplexPureImag();

            return Apply(function, real)[0];
        }
    }

    public partial class Tensor {
        /// <summary>複素純虚数</summary>
        public static Tensor ComplexPureImag(Tensor real) {
            Function function = new Functions.Complex.ComplexPureImag();

            Shape y_shape = function.OutputShapes(real.Shape)[0];

            Tensor y = new Tensor(y_shape);

            function.Execute(new Tensor[] { real }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Complex {
    /// <summary>複素純虚数</summary>
    internal class ComplexPureImag : Function {
        /// <summary>コンストラクタ</summary>
        public ComplexPureImag()
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) { }

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
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (new Tensor[] { intensor, outtensor },
                    new Operators.ComplexCast.ComplexPureImag(intensor.Shape));
        }
    }
}
