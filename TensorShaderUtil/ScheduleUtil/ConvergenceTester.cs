using System;
using System.Linq;

namespace TensorShaderUtil.ScheduleUtil {

    /// <summary>収束テスター</summary>
    /// <remarks>
    /// x方向にイテレーション，y方向に損失値をとる
    /// ピアソンの積率相関係数に基づく収束判定を行う
    /// </remarks>
    public class ConvergenceTester {
        private readonly double[] losses;
        private int pos;

        /// <summary>サンプル数</summary>
        public int Length => losses.Length;

        /// <summary>標本相関係数しきい値</summary>
        public double Threshold { private set; get; }

        /// <summary>収束判定</summary>
        public bool IsConvergenced => R() > Threshold;

        /// <summary>コンストラクタ</summary>
        /// <param name="length">サンプル数</param>
        /// <param name="threshold">標本相関係数しきい値</param>
        public ConvergenceTester(int length, double threshold) {
            if (length < 2) {
                throw new ArgumentException(nameof(length));
            }
            if (threshold <= 0 || !(threshold < 1)) {
                throw new ArgumentException(nameof(threshold));
            }

            this.losses = new double[length];
            this.Threshold = threshold;

            Initialize();
        }

        /// <summary>初期化</summary>
        public void Initialize() {
            for (int i = 0; i < losses.Length; i++) {
                losses[i] = double.NaN;
            }
            this.pos = 0;
        }

        /// <summary>追加</summary>
        /// <param name="loss">損失値</param>
        public void Next(double loss) {
            losses[pos] = loss;
            pos++;
            if (pos >= losses.Length) {
                pos = 0;
            }
        }

        /// <summary>標本相関係数</summary>
        public double R() {
            double avg_x = (losses.Length - 1) / 2.0;
            double avg_y = losses.Average();
            double sxy = 0, sx = 0, sy = 0;

            for (int i = 0, j = pos; i < losses.Length; i++, j = (j + 1) % losses.Length) {
                if (double.IsNaN(losses[j])) {
                    return double.NaN;
                }

                double dx = i - avg_x;
                double dy = losses[j] - avg_y;

                sx += dx * dx;
                sy += dy * dy;
                sxy += dx * dy;
            }

            double r = sxy / Math.Max(1e-20, Math.Sqrt(sx * sy));

            return r;
        }
    }
}
