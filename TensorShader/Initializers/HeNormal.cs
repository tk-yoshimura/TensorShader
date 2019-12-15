using System;

namespace TensorShader.Initializers {
    /// <summary>Heの初期化</summary>
    /// <remarks>
    /// He et al., https://arxiv.org/abs/1502.01852
    /// </remarks>
    public class HeNormal : LimitedNormal {
        /// <summary>コンストラクタ</summary>
        public HeNormal(Tensor tensor, Random random, float scale = 1.0f, float limitsigma = 4.0f)
            : base(tensor, random,
                  scale * (float)Math.Sqrt(2f / Fan(tensor.Shape)),
                  limitsigma) { }

        private static int Fan(Shape shape) {
            if (shape.Type != ShapeType.Kernel) {
                throw new ArgumentException(ExceptionMessage.TensorType(shape.Type, ShapeType.Kernel));
            }

            return shape.DataSize / shape.OutChannels;
        }
    }
}
