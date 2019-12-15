using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.QuaternionConvolution;

namespace TensorShaderTest.Operators.Quaternion {
    [TestClass]
    public class QuaternionKernelProduct2DTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int inchannels in new int[] { 4, 8, 12 }) {
                    foreach (int outchannels in new int[] { 4, 8, 12 }) {
                        foreach (int kheight in new int[] { 1, 3, 5 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int stride in new int[] { 1, 2, 3 }) {
                                    foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                        foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                            int outwidth = (inwidth - kwidth) / stride + 1, outheight = (inheight - kheight) / stride + 1;

                                            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                                                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

                                            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                                                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

                                            QuaternionMap2D x = new QuaternionMap2D(inchannels / 4, inwidth, inheight, batch, xcval);
                                            QuaternionMap2D y = new QuaternionMap2D(outchannels / 4, outwidth, outheight, batch, ycval);

                                            QuaternionFilter2D gw = Reference(x, y, kwidth, kheight, stride);

                                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);

                                            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels, outchannels / 4, kwidth, kheight));

                                            QuaternionKernelProduct2D ope = new QuaternionKernelProduct2D(inwidth, inheight, inchannels, outchannels, kwidth, kheight, stride, transpose: false, batch);

                                            ope.Execute(x_tensor, y_tensor, gw_tensor);

                                            float[] gw_expect = gw.ToArray();
                                            float[] gw_actual = gw_tensor.State;

                                            CollectionAssert.AreEqual(xval, x_tensor.State);
                                            CollectionAssert.AreEqual(yval, y_tensor.State);

                                            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch}");

                                            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch}");

                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void OverflowTest() {
            foreach (bool transpose in new bool[] { false, true }) {
                foreach (int batch in new int[] { 1, 2, 3 }) {
                    foreach (int inchannels in new int[] { 4, 8, 12 }) {
                        foreach (int outchannels in new int[] { 4, 8, 12 }) {
                            foreach (int kheight in new int[] { 1, 3, 5 }) {
                                foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                    foreach (int stride in new int[] { 1, 2, 3 }) {
                                        foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                            foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                                int outwidth = (inwidth - kwidth) / stride + 1, outheight = (inheight - kheight) / stride + 1;

                                                float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                                float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);

                                                OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels, outchannels / 4, kwidth, kheight));

                                                QuaternionKernelProduct2D ope = new QuaternionKernelProduct2D(inwidth, inheight, inchannels, outchannels, kwidth, kheight, stride, transpose, batch);

                                                ope.Execute(x_tensor, y_tensor, gw_tensor);

                                                CollectionAssert.AreEqual(xval, x_tensor.State);
                                                CollectionAssert.AreEqual(yval, y_tensor.State);

                                                gw_tensor.CheckOverflow();

                                                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch},{transpose}");

                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inheight = 512, inchannels = 32, outchannels = 32, ksize = 3, stride = 2;
            int outwidth = (inwidth - ksize) / stride + 1, outheight = (inheight - ksize) / stride + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels, outchannels / 4, ksize, ksize));

            QuaternionKernelProduct2D ope = new QuaternionKernelProduct2D(inwidth, inheight, inchannels, outchannels, ksize, ksize, stride);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, y_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, gw_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static QuaternionFilter2D Reference(QuaternionMap2D x, QuaternionMap2D gy, int kwidth, int kheight, int stride) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = gy.Width, outh = gy.Height;

            if (outw != (inw - kwidth) / stride + 1 || outh != (inh - kheight) / stride + 1) {
                throw new ArgumentException("mismatch shape");
            }

            QuaternionFilter2D w = new QuaternionFilter2D(inchannels, outchannels, kwidth, kheight);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int inch, outch = 0; outch < outchannels; outch++) {
                            for (inch = 0; inch < inchannels; inch++) {
                                Quaternion sum = 0;

                                for (int ix, iy = ky, ox, oy = 0; oy < outh; iy += stride, oy++) {
                                    for (ix = kx, ox = 0; ox < outw; ix += stride, ox++) {
                                        sum += Quaternion.MulGrad(gy[outch, ox, oy, th], x[inch, ix, iy, th]);
                                    }
                                }

                                w[inch, outch, kx, ky] += sum;
                            }
                        }

                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 8, outchannels = 12, kwidth = 3, kheight = 5, stride = 2, inwidth = 13, inheight = 17;
            int outwidth = (inwidth - kwidth) / stride + 1, outheight = (inheight - kheight) / stride + 1, batch = 1;

            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            QuaternionMap2D x = new QuaternionMap2D(inchannels / 4, inwidth, inheight, batch, xcval);
            QuaternionMap2D y = new QuaternionMap2D(outchannels / 4, outwidth, outheight, batch, ycval);

            QuaternionFilter2D gw = Reference(x, y, kwidth, kheight, stride);

            float[] gw_expect = {
                1.840776000e+01f,  -0.000000000e+00f,  -1.547280000e-01f,  -7.736400000e-02f,  1.857945600e+01f,  2.664535259e-15f,
                -1.554000000e-01f,  -7.770000000e-02f,  1.796054400e+01f,  -0.000000000e+00f,  -1.540560000e-01f,  -7.702800000e-02f,
                1.812955200e+01f,  -1.776356839e-15f,  -1.547280000e-01f,  -7.736400000e-02f,  1.751332800e+01f,  8.881784197e-16f,
                -1.533840000e-01f,  -7.669200000e-02f,  1.767964800e+01f,  -0.000000000e+00f,  -1.540560000e-01f,  -7.702800000e-02f,
                1.875115200e+01f,  4.440892099e-15f,  -1.560720000e-01f,  -7.803600000e-02f,  1.892284800e+01f,  -1.776356839e-15f,
                -1.567440000e-01f,  -7.837200000e-02f,  1.829856000e+01f,  8.881784197e-16f,  -1.554000000e-01f,  -7.770000000e-02f,
                1.846756800e+01f,  8.881784197e-16f,  -1.560720000e-01f,  -7.803600000e-02f,  1.784596800e+01f,  -8.881784197e-16f,
                -1.547280000e-01f,  -7.736400000e-02f,  1.801228800e+01f,  -0.000000000e+00f,  -1.554000000e-01f,  -7.770000000e-02f,
                1.909454400e+01f,  -8.881784197e-16f,  -1.574160000e-01f,  -7.870800000e-02f,  1.926624000e+01f,  -0.000000000e+00f,
                -1.580880000e-01f,  -7.904400000e-02f,  1.863657600e+01f,  -8.881784197e-16f,  -1.567440000e-01f,  -7.837200000e-02f,
                1.880558400e+01f,  8.881784197e-16f,  -1.574160000e-01f,  -7.870800000e-02f,  1.817860800e+01f,  -8.881784197e-16f,
                -1.560720000e-01f,  -7.803600000e-02f,  1.834492800e+01f,  -0.000000000e+00f,  -1.567440000e-01f,  -7.837200000e-02f,
                2.287185600e+01f,  -5.329070518e-15f,  -1.722000000e-01f,  -8.610000000e-02f,  2.304355200e+01f,  8.881784197e-16f,
                -1.728720000e-01f,  -8.643600000e-02f,  2.235475200e+01f,  7.105427358e-15f,  -1.715280000e-01f,  -8.576400000e-02f,
                2.252376000e+01f,  1.776356839e-15f,  -1.722000000e-01f,  -8.610000000e-02f,  2.183764800e+01f,  -0.000000000e+00f,
                -1.708560000e-01f,  -8.542800000e-02f,  2.200396800e+01f,  -0.000000000e+00f,  -1.715280000e-01f,  -8.576400000e-02f,
                2.321524800e+01f,  2.664535259e-15f,  -1.735440000e-01f,  -8.677200000e-02f,  2.338694400e+01f,  -0.000000000e+00f,
                -1.742160000e-01f,  -8.710800000e-02f,  2.269276800e+01f,  -1.776356839e-15f,  -1.728720000e-01f,  -8.643600000e-02f,
                2.286177600e+01f,  -1.776356839e-15f,  -1.735440000e-01f,  -8.677200000e-02f,  2.217028800e+01f,  -0.000000000e+00f,
                -1.722000000e-01f,  -8.610000000e-02f,  2.233660800e+01f,  -8.881784197e-16f,  -1.728720000e-01f,  -8.643600000e-02f,
                2.355864000e+01f,  2.664535259e-15f,  -1.748880000e-01f,  -8.744400000e-02f,  2.373033600e+01f,  -2.664535259e-15f,
                -1.755600000e-01f,  -8.778000000e-02f,  2.303078400e+01f,  -3.552713679e-15f,  -1.742160000e-01f,  -8.710800000e-02f,
                2.319979200e+01f,  1.776356839e-15f,  -1.748880000e-01f,  -8.744400000e-02f,  2.250292800e+01f,  -0.000000000e+00f,
                -1.735440000e-01f,  -8.677200000e-02f,  2.266924800e+01f,  -0.000000000e+00f,  -1.742160000e-01f,  -8.710800000e-02f,
                2.733595200e+01f,  -8.881784197e-16f,  -1.896720000e-01f,  -9.483600000e-02f,  2.750764800e+01f,  -2.664535259e-15f,
                -1.903440000e-01f,  -9.517200000e-02f,  2.674896000e+01f,  -0.000000000e+00f,  -1.890000000e-01f,  -9.450000000e-02f,
                2.691796800e+01f,  -2.664535259e-15f,  -1.896720000e-01f,  -9.483600000e-02f,  2.616196800e+01f,  8.881784197e-16f,
                -1.883280000e-01f,  -9.416400000e-02f,  2.632828800e+01f,  -8.881784197e-16f,  -1.890000000e-01f,  -9.450000000e-02f,
                2.767934400e+01f,  3.552713679e-15f,  -1.910160000e-01f,  -9.550800000e-02f,  2.785104000e+01f,  -1.776356839e-15f,
                -1.916880000e-01f,  -9.584400000e-02f,  2.708697600e+01f,  -8.881784197e-16f,  -1.903440000e-01f,  -9.517200000e-02f,
                2.725598400e+01f,  1.776356839e-15f,  -1.910160000e-01f,  -9.550800000e-02f,  2.649460800e+01f,  -0.000000000e+00f,
                -1.896720000e-01f,  -9.483600000e-02f,  2.666092800e+01f,  -1.776356839e-15f,  -1.903440000e-01f,  -9.517200000e-02f,
                2.802273600e+01f,  8.881784197e-16f,  -1.923600000e-01f,  -9.618000000e-02f,  2.819443200e+01f,  2.664535259e-15f,
                -1.930320000e-01f,  -9.651600000e-02f,  2.742499200e+01f,  -0.000000000e+00f,  -1.916880000e-01f,  -9.584400000e-02f,
                2.759400000e+01f,  4.440892099e-15f,  -1.923600000e-01f,  -9.618000000e-02f,  2.682724800e+01f,  -0.000000000e+00f,
                -1.910160000e-01f,  -9.550800000e-02f,  2.699356800e+01f,  -1.776356839e-15f,  -1.916880000e-01f,  -9.584400000e-02f,
                3.180004800e+01f,  3.552713679e-15f,  -2.071440000e-01f,  -1.035720000e-01f,  3.197174400e+01f,  -4.440892099e-15f,
                -2.078160000e-01f,  -1.039080000e-01f,  3.114316800e+01f,  -0.000000000e+00f,  -2.064720000e-01f,  -1.032360000e-01f,
                3.131217600e+01f,  -3.552713679e-15f,  -2.071440000e-01f,  -1.035720000e-01f,  3.048628800e+01f,  -8.881784197e-16f,
                -2.058000000e-01f,  -1.029000000e-01f,  3.065260800e+01f,  -0.000000000e+00f,  -2.064720000e-01f,  -1.032360000e-01f,
                3.214344000e+01f,  -1.776356839e-15f,  -2.084880000e-01f,  -1.042440000e-01f,  3.231513600e+01f,  -7.105427358e-15f,
                -2.091600000e-01f,  -1.045800000e-01f,  3.148118400e+01f,  2.664535259e-15f,  -2.078160000e-01f,  -1.039080000e-01f,
                3.165019200e+01f,  -8.881784197e-16f,  -2.084880000e-01f,  -1.042440000e-01f,  3.081892800e+01f,  -0.000000000e+00f,
                -2.071440000e-01f,  -1.035720000e-01f,  3.098524800e+01f,  -0.000000000e+00f,  -2.078160000e-01f,  -1.039080000e-01f,
                3.248683200e+01f,  3.552713679e-15f,  -2.098320000e-01f,  -1.049160000e-01f,  3.265852800e+01f,  -1.776356839e-15f,
                -2.105040000e-01f,  -1.052520000e-01f,  3.181920000e+01f,  8.881784197e-16f,  -2.091600000e-01f,  -1.045800000e-01f,
                3.198820800e+01f,  -2.664535259e-15f,  -2.098320000e-01f,  -1.049160000e-01f,  3.115156800e+01f,  -0.000000000e+00f,
                -2.084880000e-01f,  -1.042440000e-01f,  3.131788800e+01f,  8.881784197e-16f,  -2.091600000e-01f,  -1.045800000e-01f,
                3.626414400e+01f,  -0.000000000e+00f,  -2.246160000e-01f,  -1.123080000e-01f,  3.643584000e+01f,  -1.776356839e-15f,
                -2.252880000e-01f,  -1.126440000e-01f,  3.553737600e+01f,  1.776356839e-15f,  -2.239440000e-01f,  -1.119720000e-01f,
                3.570638400e+01f,  7.105427358e-15f,  -2.246160000e-01f,  -1.123080000e-01f,  3.481060800e+01f,  3.552713679e-15f,
                -2.232720000e-01f,  -1.116360000e-01f,  3.497692800e+01f,  1.776356839e-15f,  -2.239440000e-01f,  -1.119720000e-01f,
                3.660753600e+01f,  -7.105427358e-15f,  -2.259600000e-01f,  -1.129800000e-01f,  3.677923200e+01f,  -3.552713679e-15f,
                -2.266320000e-01f,  -1.133160000e-01f,  3.587539200e+01f,  -1.776356839e-15f,  -2.252880000e-01f,  -1.126440000e-01f,
                3.604440000e+01f,  -8.881784197e-15f,  -2.259600000e-01f,  -1.129800000e-01f,  3.514324800e+01f,  -0.000000000e+00f,
                -2.246160000e-01f,  -1.123080000e-01f,  3.530956800e+01f,  -8.881784197e-15f,  -2.252880000e-01f,  -1.126440000e-01f,
                3.695092800e+01f,  -3.552713679e-15f,  -2.273040000e-01f,  -1.136520000e-01f,  3.712262400e+01f,  1.776356839e-15f,
                -2.279760000e-01f,  -1.139880000e-01f,  3.621340800e+01f,  -3.552713679e-15f,  -2.266320000e-01f,  -1.133160000e-01f,
                3.638241600e+01f,  -5.329070518e-15f,  -2.273040000e-01f,  -1.136520000e-01f,  3.547588800e+01f,  -0.000000000e+00f,
                -2.259600000e-01f,  -1.129800000e-01f,  3.564220800e+01f,  -3.552713679e-15f,  -2.266320000e-01f,  -1.133160000e-01f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch}"); /*many fma tolerance*/
        }
    }
}
