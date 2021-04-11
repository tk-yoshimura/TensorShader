using System.Collections.Generic;
using System.Linq;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>平均</summary>
        public static VariableNode Average(VariableNode x, int[] axes = null, bool keepdims = false) {
            if (axes is null) {
                axes = (new int[x.Shape.Ndim]).Select((_, dim) => dim).ToArray();
            }

            foreach (int axis in axes.OrderBy((val) => val)) {
                if (x.Shape[axis] == 1) continue;

                Function function = new Functions.Aggregation.Average(axis);

                x = Apply(function, x)[0];
            }

            if (!keepdims) {
                List<int> lengths = ((int[])x.Shape).ToList();

                foreach (int axis in axes.OrderByDescending((val) => val)) {
                    lengths.RemoveAt(axis);
                }

                x = Reshape(x, new Shape(x.Shape.Type, lengths.ToArray()));
            }

            return x;
        }
    }

    public partial class Tensor {
        /// <summary>平均</summary>
        public static Tensor Average(Tensor x, int[] axes = null, bool keepdims = false) {
            if (axes is null) {
                axes = (new int[x.Shape.Ndim]).Select((_, dim) => dim).ToArray();
            }

            foreach (int axis in axes.OrderBy((val) => val)) {
                if (x.Shape[axis] == 1) continue;

                Function function = new Functions.Aggregation.Average(axis);

                Tensor y = new(function.OutputShapes(x.Shape)[0]);

                function.Execute(new Tensor[] { x }, new Tensor[] { y });

                x = y;
            }

            if (!keepdims) {
                List<int> lengths = ((int[])x.Shape).ToList();

                foreach (int axis in axes.OrderByDescending((val) => val)) {
                    lengths.RemoveAt(axis);
                }

                x = Reshape(x, new Shape(x.Shape.Type, lengths.ToArray()));
            }

            return x;
        }
    }
}

namespace TensorShader.Functions.Aggregation {
    /// <summary>平均</summary>
    internal class Average : Aggregation {
        /// <summary>コンストラクタ</summary>
        public Average(int axis)
            : base(axis) { }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (new Tensor[] { intensor, outtensor }, new Operators.Aggregation.Average(intensor.Shape, Axis));
        }
    }
}
