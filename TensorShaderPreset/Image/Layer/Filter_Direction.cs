using System;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderPreset.Image {

    /// <summary>フィルタ</summary>
    public static partial class Filter {

        private static NdimArray<float> KernelDirection(double cosdir, double sindir, int ksize, double smoothness) {
            double g(double x, double y) {
                return Math.Pow(2, -smoothness * x * x - 4 * y * y);
            }

            double h(double x, double y) {
                return g(x, y - 1) - g(x, y + 1);
            }

            double scale = 0;
            double[] vs = new double[ksize * ksize];
            for (int i = 0, y = -(ksize / 2); y <= ksize / 2; y++) {
                for (int x = -(ksize / 2); x <= ksize / 2; x++, i++) {
                    double u = x * cosdir + y * sindir;
                    double v = y * cosdir - x * sindir;

                    vs[i] = h(u, v);
                    scale += Math.Max(0, vs[i]);
                }
            }

            float[] wval = vs.Select((v) => (float)(v / scale)).ToArray();

            return new NdimArray<float>(Shape.Kernel2D(1, 1, ksize, ksize), wval);
        }

        /// <summary>2冪数方向微分フィルタ</summary>
        /// <param name="dirs">方向分割数(2冪数かつ4以上256以下)</param>
        /// <param name="ksize">フィルタサイズ</param>
        /// <param name="smoothness">フィルタの方向滑らかさ</param>
        public static NdimArray<float> KernelDirection(int dirs, int ksize = 5, double smoothness = 0.5) {
            if (dirs < 4 || dirs > 256 || (dirs & (dirs - 1)) != 0) {
                throw new ArgumentException(nameof(dirs));
            }

            (double[] c, double[] s) cossindir(int divdirs) {
                double[] c = { 1, 0, -1 }, s = { 0, 1, 0 };
                int n = 3;

                while (divdirs > 2) {
                    int new_n = n * 2 - 1;
                    double[] new_c = new double[new_n], new_s = new double[new_n];

                    for (int i = 0; i < n; i++) {
                        new_c[i * 2] = c[i];
                        new_s[i * 2] = s[i];
                    }

                    for (int i = 1; i < n; i++) {
                        double ca = c[i - 1], cb = c[i], sa = s[i - 1], sb = s[i];
                        double c2 = ca * cb - sa * sb;

                        new_c[i * 2 - 1] = ((i * 2 < n) ? 1 : -1) * Math.Sqrt((1 + c2) / 2);
                        new_s[i * 2 - 1] = Math.Sqrt((1 - c2) / 2);
                    }

                    n = new_n; c = new_c; s = new_s;

                    divdirs /= 2;
                }

                c = c.Take(c.Length - 1).ToArray();
                s = s.Take(s.Length - 1).ToArray();

                return (c, s);
            }

            (double[] cosdirs, double[] sindirs) = cossindir(dirs);

            NdimArray<float>[] arrays = new NdimArray<float>[dirs];

            for (int i = 0; i < arrays.Length; i++) {
                arrays[i] = KernelDirection(cosdirs[i], sindirs[i], ksize, smoothness);
            }

            return NdimArray<float>.Join(Axis.Kernel2D.OutChannels, arrays);
        }

        /// <summary>2冪数方向微分フィルタ</summary>
        /// <param name="x">入力フィールド(単チャネル)</param>
        /// <param name="dirs">方向分割数(2冪数かつ4以上256以下)</param>
        /// <param name="ksize">フィルタサイズ</param>
        /// <param name="smoothness">フィルタの方向滑らかさ</param>
        public static Field Direction(Field x, int dirs, int ksize = 5, double smoothness = 0.5) {
            VariableField w = new VariableField(
                KernelDirection(dirs, ksize, smoothness),
                name: $"Direction{dirs}"
            );

            return Convolution2D(x, w);
        }
    }
}
