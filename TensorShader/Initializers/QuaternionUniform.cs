using System;

namespace TensorShader.Initializers {
    /// <summary>重み四元数の初期化</summary>
    public class QuaternionUniform : Uniform {
        /// <summary>コンストラクタ</summary>
        public QuaternionUniform(Tensor tensor, Random random, float scale = 1.0f)
            : base(tensor, random,
                  (float)(scale / EqGainCoef(tensor.Shape))) { }

        /// <summary>
        /// 四元数の要素が分散1の対称一様分布に従うとき、
        /// 四元数との積の総和の分散が1となる四元数の標準偏差の逆数
        /// </summary>
        private static double EqGainCoef(Shape shape) {
            if (shape.Type != ShapeType.Kernel) {
                throw new ArgumentException(ExceptionMessage.ShapeType(shape.Type, ShapeType.Kernel));
            }

            int inunits = shape.DataSize / shape.OutChannels;

            return Math.Sqrt(inunits / 3.0); // (4 / 3 * N)^(1/2) N : number of quaternion
        }
    }
}
