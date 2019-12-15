using System;

namespace TensorShader.Initializers {
    /// <summary>3次元ベクトル畳み込み層の重み四元数の初期化</summary>
    public class TrivectorUniform : Uniform {
        /// <summary>コンストラクタ</summary>
        public TrivectorUniform(Tensor tensor, Random random, float scale = 1.0f)
            : base(tensor, random,
                  (float)(Math.Sqrt(scale) / EqGainCoef(tensor.Shape))) { }

        /// <summary>
        /// 3次元ベクトルの要素が分散1の対称一様分布に従うとき、
        /// 四元数との回転積の総和の分散が1となる四元数の標準偏差の逆数
        /// </summary>
        private static double EqGainCoef(Shape shape) {
            if (shape.Type != ShapeType.Kernel) {
                throw new ArgumentException(ExceptionMessage.TensorType(shape.Type, ShapeType.Kernel));
            }

            int inunits = shape.DataSize / shape.OutChannels;

            return Math.Sqrt(Math.Sqrt(inunits * 8.0 / 15.0)); // (32 / 15 * N)^(1/4) N : number of 3D-vectors
        }
    }
}
