using System;

namespace TensorShader.Initializers {
    /// <summary>重み複素数の初期化</summary>
    public class ComplexNormal : LimitedNormal {
        /// <summary>コンストラクタ</summary>
        public ComplexNormal(Tensor tensor, Random random, float scale = 1.0f, float limitsigma = 4.0f)
            : base(tensor, random,
                  (float)(scale / EqGainCoef(tensor.Shape)),
                  limitsigma) { }

        /// <summary>
        /// 複素数の要素が標準正規分布に従うとき、
        /// 複素数との積の総和の分散が1となる複素数の標準偏差の逆数
        /// </summary>
        private static double EqGainCoef(Shape shape) {
            if (shape.Type != ShapeType.Kernel) {
                throw new ArgumentException(ExceptionMessage.ShapeType(shape.Type, ShapeType.Kernel));
            }

            int inunits = shape.DataSize / shape.OutChannels;

            return Math.Sqrt(inunits); // (2 * N)^(1/2) N : number of complex
        }
    }
}
