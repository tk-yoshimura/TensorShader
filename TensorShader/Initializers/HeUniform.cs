using System;

namespace TensorShader.Initializers {
    /// <summary>Heの初期化</summary>
    /// <remarks>
    /// He et al., https://arxiv.org/abs/1502.01852
    /// </remarks>
    public class HeUniform : Uniform {
        /// <summary>コンストラクタ</summary>
        public HeUniform(Tensor tensor, Random random, float scale = 1.0f)
            : base(tensor, random,
                  scale * (float)Math.Sqrt(6f / Fan(tensor.Shape))) { }

        private static int Fan(Shape shape) {
            if (shape.Type != ShapeType.Kernel) {
                throw new ArgumentException(ExceptionMessage.ShapeType(shape.Type, ShapeType.Kernel));
            }

            return shape.DataSize / shape.OutChannels;
        }
    }
}
