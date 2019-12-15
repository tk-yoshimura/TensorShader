using System;

namespace TensorShader.Initializers {
    /// <summary>Glorotの初期化</summary>
    /// <remarks>
    /// Glorot, Bengio, AISTATS 2010
    /// </remarks>
    public class GlorotUniform : Uniform {
        /// <summary>コンストラクタ</summary>
        public GlorotUniform(Tensor tensor, Random random, float scale = 1.0f)
            : base(tensor, random,
                  scale * (float)Math.Sqrt(6f / Fan(tensor.Shape))) { }

        private static int Fan(Shape shape) {
            if (shape.Type != ShapeType.Kernel) {
                throw new ArgumentException(ExceptionMessage.TensorType(shape.Type, ShapeType.Kernel));
            }

            return shape.DataSize / shape.OutChannels + shape.OutChannels;
        }
    }
}
