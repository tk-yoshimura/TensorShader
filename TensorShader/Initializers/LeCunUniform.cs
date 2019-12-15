using System;

namespace TensorShader.Initializers {
    /// <summary>LeCunの初期化</summary>
    /// <remarks>
    /// LeCun 98, Efficient Backprop
    /// http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    /// </remarks>
    public class LeCunUniform : Uniform {
        /// <summary>コンストラクタ</summary>
        public LeCunUniform(Tensor tensor, Random random, float scale = 1.0f)
            : base(tensor, random,
                  scale * (float)Math.Sqrt(3f / Fan(tensor.Shape))) { }

        private static int Fan(Shape shape) {
            if (shape.Type != ShapeType.Kernel) {
                throw new ArgumentException(ExceptionMessage.TensorType(shape.Type, ShapeType.Kernel));
            }

            return shape.DataSize / shape.OutChannels;
        }
    }
}
