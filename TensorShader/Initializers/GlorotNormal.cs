using System;

namespace TensorShader.Initializers {
    /// <summary>Glorotの初期化</summary>
    /// <remarks>
    /// Glorot, Bengio, AISTATS 2010
    /// </remarks>
    public class GlorotNormal : LimitedNormal {
        /// <summary>コンストラクタ</summary>
        public GlorotNormal(Tensor tensor, Random random, float scale = 1.0f, float limitsigma = 4.0f)
            : base(tensor, random,
                  scale * (float)Math.Sqrt(2f / Fan(tensor.Shape)),
                  limitsigma) { }

        private static int Fan(Shape shape) {
            if (shape.Type != ShapeType.Kernel) {
                throw new ArgumentException(ExceptionMessage.TensorType(shape.Type, ShapeType.Kernel));
            }

            return shape.DataSize / shape.OutChannels + shape.OutChannels;
        }
    }
}
