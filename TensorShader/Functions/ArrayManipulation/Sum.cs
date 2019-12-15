using System;
using System.Linq;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>テンソル総和</summary>
        public static VariableNode Sum(params VariableNode[] xs) {
            if (xs.Length == 1) {
                return xs[0];
            }
            else if (xs.Length == 2) {
                return xs[0] + xs[1];
            }

            Function function = new Functions.ArrayManipulation.Sum(xs.Length);

            return Apply(function, xs)[0];
        }
    }

    public partial class Tensor {
        /// <summary>テンソル総和</summary>
        public static Tensor Sum(params Tensor[] xs) {
            if (xs.Length == 1) {
                return xs[0];
            }
            else if (xs.Length == 2) {
                return xs[0] + xs[1];
            }

            Function function = new Functions.ArrayManipulation.Sum(xs.Length);

            Tensor y = new Tensor(function.OutputShapes(xs.Select((tensor) => tensor.Shape).ToArray())[0]);

            function.Execute(xs, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.ArrayManipulation {
    /// <summary>テンソル総和</summary>
    internal class Sum : Function {
        /// <summary>テンソル総和</summary>
        public Sum(int inputs)
            : base(inputs, outputs: 1, allow_resubstitution: false) { }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return new Shape[] { inshapes[0] };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            for (int i = 1; i < inshapes.Length; i++) {
                if (inshapes[i] != inshapes[0]) {
                    throw new ArgumentException(ExceptionMessage.ShapeWithIndex(index: i, inshapes[i], inshapes[0]));
                }
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor outtensor = outtensors[0];

            return (
                intensors.Concat(outtensors).ToArray(),
                new Operators.ArrayManipulation.Sum(
                    outtensor.Shape,
                    intensors.Length)
                );
        }
    }
}
