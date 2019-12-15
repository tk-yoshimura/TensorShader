using System;

namespace TensorShader.Initializers {
    /// <summary>LeCunの初期化</summary>
    /// <remarks>
    /// LeCun 98, Efficient Backprop
    /// http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    /// </remarks>
    public class LeCunNormal : LimitedNormal {
        /// <summary>コンストラクタ</summary>
        public LeCunNormal(Tensor tensor, Random random, float scale = 1.0f, float limitsigma = 4.0f)
            : base(tensor, random,
                  scale * (float)Math.Sqrt(1f / Fan(tensor.Shape)),
                  limitsigma) { }

        private static int Fan(Shape shape) {
            if (shape.Type != ShapeType.Kernel) {
                throw new ArgumentException(ExceptionMessage.TensorType(shape.Type, ShapeType.Kernel));
            }

            return shape.DataSize / shape.OutChannels;
        }
    }
}
