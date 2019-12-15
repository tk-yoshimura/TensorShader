using System;
using System.Linq;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>結合</summary>
        public static VariableNode Concat(int axis, params VariableNode[] xs) {
            Function function = new Functions.ArrayManipulation.Concat(xs.Length, axis);

            return Apply(function, xs)[0];
        }
    }

    public partial class Tensor {
        /// <summary>結合</summary>
        public static Tensor Concat(int axis, params Tensor[] xs) {
            Function function = new Functions.ArrayManipulation.Concat(xs.Length, axis);

            Tensor y = new Tensor(function.OutputShapes(xs.Select((tensor) => tensor.Shape).ToArray())[0]);

            function.Execute(xs, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.ArrayManipulation {
    /// <summary>結合</summary>
    internal class Concat : Function {
        /// <summary>軸</summary>
        public int Axis { private set; get; }

        /// <summary>結合</summary>
        public Concat(int inputs, int axis)
            : base(inputs, outputs: 1, allow_resubstitution: false) {
            this.Axis = axis;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            int axislength = inshapes.Select((shape) => shape[Axis]).Sum();

            int[] s = inshapes[0];
            s[Axis] = axislength;

            Shape outshape = new Shape(inshapes[0].Type, s);

            return new Shape[] { outshape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes.Select((shape) => shape.Ndim).Any((ndim) => ndim <= Axis)) {
                throw new ArgumentOutOfRangeException(nameof(Axis));
            }

            for (int i = 1; i < inshapes.Length; i++) {
                if (inshapes[i].Ndim != inshapes[0].Ndim) {
                    throw new ArgumentException(ExceptionMessage.Concat(Axis, inshapes));
                }

                for (int j = 0; j < inshapes[0].Ndim; j++) {
                    if (j == Axis) continue;

                    if (inshapes[i][j] != inshapes[0][j]) {
                        throw new ArgumentException(ExceptionMessage.Concat(Axis, inshapes));
                    }
                }
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (
                intensors.Concat(outtensors).ToArray(),
                new Operators.ArrayManipulation.Concat(
                    intensors.Select((tensor) => tensor.Shape).ToArray(),
                    outtensor.Shape,
                    Axis)
                );
        }
    }
}
