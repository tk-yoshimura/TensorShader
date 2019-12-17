using System;
using System.Linq;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>分離</summary>
        public static VariableNode[] Separate(VariableNode x, int axis, int[] lengths) {
            Function function = new Functions.ArrayManipulation.Separate(axis, lengths);

            return Apply(function, x);
        }
    }

    public partial class Tensor {
        /// <summary>分離</summary>
        public static Tensor[] Separate(Tensor x, int axis, int[] lengths) {
            Function function = new Functions.ArrayManipulation.Separate(axis, lengths);

            Tensor[] ys = function.OutputShapes(x.Shape).Select((shape) => new Tensor(shape)).ToArray();

            function.Execute(new Tensor[] { x }, ys);

            return ys;
        }
    }
}

namespace TensorShader.Functions.ArrayManipulation {
    /// <summary>分離</summary>
    internal class Separate : Function {
        private readonly int[] lengths;

        /// <summary>軸</summary>
        public int Axis { private set; get; }

        /// <summary>分離長さリスト</summary>
        public int[] Lengths => (int[])lengths.Clone();

        /// <summary>コンストラクタ</summary>
        public Separate(int axis, int[] lengths)
            : base(inputs: 1, outputs: lengths.Length, allow_resubstitution: false) {

            this.lengths = (int[])lengths.Clone();
            this.Axis = axis;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            int[] s = inshape;

            Shape[] outshapes =
                lengths.Select((length) => {
                    int[] outshape = (int[])s.Clone();
                    outshape[Axis] = length;
                    return new Shape(inshape.Type, outshape);
                }).ToArray();

            return outshapes;
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0][Axis] != Lengths.Sum()) {
                throw new ArgumentException(ExceptionMessage.Shape("AxisLength", inshapes[0]));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0];

            return (
                intensors.Concat(outtensors).ToArray(),
                new Operators.ArrayManipulation.Separate(
                    intensor.Shape,
                    outtensors.Select((tensor) => tensor.Shape).ToArray(),
                    Axis)
                );
        }
    }
}
