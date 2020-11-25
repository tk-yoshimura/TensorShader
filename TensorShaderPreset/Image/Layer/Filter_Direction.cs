using System;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderPreset.Image {

    /// <summary>フィルタ</summary>
    public static partial class Filter {

        private static NdimArray<float> KernelDirection(double cosdir, double sindir, int ksize, double smooth) {
            double g(double x, double y) {
                return Math.Pow(2, -smooth * x * x - 4 * y * y);
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

        /// <summary>8方向微分フィルタ</summary>
        public static NdimArray<float> KernelDirection8(int ksize = 5, double smooth = 0.5) {
            double[] cosdirs = { 
                1.0, +0.9238795325112867, +0.7071067811865475, +0.3826834323650898, 
                0.0, -0.3826834323650898, -0.7071067811865475, -0.9238795325112867 
            };

            double[] sindirs = { 
                0.0, 0.3826834323650898, 0.7071067811865475, 0.9238795325112867, 
                1.0, 0.9238795325112867, 0.7071067811865475, 0.3826834323650898 
            };

            NdimArray<float>[] arrays = new NdimArray<float>[8];

            for (int i = 0; i < arrays.Length; i++) {
                arrays[i] = KernelDirection(cosdirs[i], sindirs[i], ksize, smooth);
            }

            return NdimArray<float>.Join(Axis.Kernel2D.OutChannels, arrays);
        }

        /// <summary>8方向微分フィルタ</summary>
        public static Field Direction8(Field x, int ksize = 5, double smooth = 0.5) {
            VariableField w = new VariableField(
                KernelDirection8(ksize, smooth),
                name: "Direction8"
            );

            return Convolution2D(x, w);
        }

        /// <summary>16方向微分フィルタ</summary>
        public static NdimArray<float> KernelDirection16(int ksize = 5, double smooth = 0.5) {
            double[] cosdirs = {
                1.0, +0.9807852804032304, +0.9238795325112867, +0.8314696123025452,
                +0.7071067811865475, +0.5555702330196022, +0.3826834323650898, +0.1950903220161282,
                0.0, -0.1950903220161282, -0.3826834323650898, -0.5555702330196022,
                -0.7071067811865475, -0.8314696123025452, -0.9238795325112867, -0.9807852804032304 
            };

            double[] sindirs = { 
                0.0, 0.1950903220161282, 0.3826834323650898, 0.5555702330196022,
                0.7071067811865475, 0.8314696123025452, 0.9238795325112867, 0.9807852804032304,
                1.0, 0.9807852804032304, 0.9238795325112867, 0.8314696123025452,
                0.7071067811865475, 0.5555702330196022, 0.3826834323650898, 0.1950903220161282 
            };

            NdimArray<float>[] arrays = new NdimArray<float>[16];

            for (int i = 0; i < arrays.Length; i++) {
                arrays[i] = KernelDirection(cosdirs[i], sindirs[i], ksize, smooth);
            }

            return NdimArray<float>.Join(Axis.Kernel2D.OutChannels, arrays);
        }

        /// <summary>16方向微分フィルタ</summary>
        public static Field Direction16(Field x, int ksize = 5, double smooth = 0.5) {
            VariableField w = new VariableField(
                KernelDirection16(ksize, smooth),
                name: "Direction16"
            );

            return Convolution2D(x, w);
        }
    }
}
