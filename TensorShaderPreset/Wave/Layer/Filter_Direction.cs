using System;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderPreset.Wave {

    /// <summary>フィルタ</summary>
    public static partial class Filter {

        /// <summary>ハミング窓の有効周波数範囲</summary>
        public static (double hz_min, double hz_max) HammingWindowRange(int ksize, double dt) {
            if ((ksize & 1) != 1 || ksize < 9) {
                throw new ArgumentException(nameof(ksize));
            }
            if (!(dt > 0)) {
                throw new ArgumentOutOfRangeException(nameof(dt));
            }

            double min = 2 / (ksize * dt);
            double max = 1 / (4 * dt);

            return (min, max);
        }

        /// <summary>ハミング窓</summary>
        /// <param name="ksize">窓サイズ</param>
        /// <param name="dt">サンプリング間隔[s]</param>
        /// <param name="hz">ピーク周波数[Hz]</param>
        /// <returns>余弦波、正弦波</returns>
        public static (NdimArray<float> cos, NdimArray<float> sin) KernelHammingWindow(int ksize, double dt, double hz) {
            if ((ksize & 1) != 1) {
                throw new ArgumentException(nameof(ksize));
            }
            if (!(dt > 0)) {
                throw new ArgumentOutOfRangeException(nameof(dt));
            }
            if (!(hz > 0)) {
                throw new ArgumentOutOfRangeException(nameof(hz));
            }
            if (dt * hz * 4 > 1) {
                throw new ArgumentException($"Nyquist frequency limit : {nameof(dt)} * {nameof(hz)} <= 1 / 4");
            }
            if (ksize < 9 || ksize * dt * hz < 2) {
                throw new ArgumentException($"Insufficient {nameof(ksize)} : {nameof(ksize)} >= 9 and {nameof(ksize)} * {nameof(dt)} * {nameof(hz)} >= 2");
            }

            int m = ksize / 2;

            double hamming_weight(double x) {
                return 0.54 + 0.46 * Math.Cos(Math.PI * x);
            }

            (double c, double s) cs(double x) {
                double t = 2 * x * m * Math.PI * dt * hz;

                return (Math.Cos(t), Math.Sin(t));
            }

            double h = 1.0 / m, sum_c = 0, sum_s = 0;
            double[] cosw = new double[ksize], sinw = new double[ksize];

            for (int i = 0, j = -m; i < ksize; i++, j++) {
                double x = j * h;
                double w = hamming_weight(x);
                (double c, double s) = cs(x);

                cosw[i] = c * w; sinw[i] = s * w;
                sum_c += c * c * w; sum_s += s * s * w;
            }

            float[] cosn = cosw.Select((v) => (float)(v / sum_c)).ToArray();
            float[] sinn = sinw.Select((v) => (float)(v / sum_s)).ToArray();

            return ((Shape.Kernel1D(1, 1, ksize), cosn), (Shape.Kernel1D(1, 1, ksize), sinn));
        }

        /// <summary>ハミング窓</summary>
        /// <param name="ksize">窓サイズ</param>
        /// <param name="dt">サンプリング間隔[s]</param>
        /// <param name="hz_list">ピーク周波数リスト[Hz]</param>
        /// <returns>余弦波、正弦波</returns>
        public static (NdimArray<float> cos, NdimArray<float> sin) KernelHammingWindow(int ksize, double dt, double[] hz_list) {
            NdimArray<float>[] cos_list = new NdimArray<float>[hz_list.Length];
            NdimArray<float>[] sin_list = new NdimArray<float>[hz_list.Length];

            for (int i = 0; i < hz_list.Length; i++) {
                (cos_list[i], sin_list[i]) = KernelHammingWindow(ksize, dt, hz_list[i]);
            }

            NdimArray<float> cos = NdimArray<float>.Join(Axis.Kernel1D.OutChannels, cos_list);
            NdimArray<float> sin = NdimArray<float>.Join(Axis.Kernel1D.OutChannels, sin_list);

            return (cos, sin);
        }

        /// <summary>ハミング窓</summary>
        /// <param name="x">入力フィールド(単チャネル)</param>
        /// <param name="ksize">窓サイズ</param>
        /// <param name="dt">サンプリング間隔[s]</param>
        /// <param name="hz">ピーク周波数[Hz]</param>
        /// <returns>余弦波、正弦波</returns>
        public static (Field ycos, Field ysin) HammingWindow(Field x, int ksize, double dt, double hz) {
            return HammingWindow(x, ksize, dt, new double[] { hz });
        }

        /// <summary>ハミング窓</summary>
        /// <param name="x">入力フィールド(単チャネル)</param>
        /// <param name="ksize">窓サイズ</param>
        /// <param name="dt">サンプリング間隔[s]</param>
        /// <param name="hz_list">ピーク周波数リスト[Hz]</param>
        /// <returns>余弦波、正弦波</returns>
        public static (Field ycos, Field ysin) HammingWindow(Field x, int ksize, double dt, double[] hz_list) {
            (NdimArray<float> cos, NdimArray<float> sin) = KernelHammingWindow(ksize, dt, hz_list);

            VariableField cosw = new(
                cos,
                name: $"CosHammingWindow"
            );

            VariableField sinw = new(
                sin,
                name: $"SinHammingWindow"
            );

            return (Convolution1D(x, cosw), Convolution1D(x, sinw));
        }

        /// <summary>パワースペクトル</summary>
        /// <param name="x">入力フィールド(単チャネル)</param>
        /// <param name="ksize">窓サイズ</param>
        /// <param name="dt">サンプリング間隔[s]</param>
        /// <param name="hz">ピーク周波数[Hz]</param>
        public static Field PSD(Field x, int ksize, double dt, double hz) {
            return PSD(x, ksize, dt, new double[] { hz });
        }

        /// <summary>パワースペクトル</summary>
        /// <param name="x">入力フィールド(単チャネル)</param>
        /// <param name="ksize">窓サイズ</param>
        /// <param name="dt">サンプリング間隔[s]</param>
        /// <param name="hz_list">ピーク周波数リスト[Hz]</param>
        public static Field PSD(Field x, int ksize, double dt, double[] hz_list) {
            (Field ycos, Field ysin) = HammingWindow(x, ksize, dt, hz_list);

            return Square(ycos) + Square(ysin);
        }
    }
}
