using System.Collections.Generic;

namespace TensorShader.Operators.Aggregation {
    /// <summary>次元ごとに集計</summary>
    internal abstract class Aggregation : Operator {
        protected readonly uint Stride, AxisLength, Slides;

        /// <summary>コンストラクタ</summary>
        protected Aggregation(Shape shape, int axis) {
            int[] new_shape = shape;
            new_shape[axis] = 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, shape),
                (ArgumentType.Out, new Shape(shape.Type, new_shape)),
            };

            AxisLength = (uint)shape[axis];
            Stride = Slides = 1;

            for (int i = 0; i < axis; i++) {
                Stride *= (uint)shape[i];
            }

            for (int i = axis + 1; i < shape.Ndim; i++) {
                Slides *= (uint)shape[i];
            }
        }
    }
}
