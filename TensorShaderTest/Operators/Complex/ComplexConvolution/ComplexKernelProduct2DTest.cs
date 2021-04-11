using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.ComplexConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Complex {
    [TestClass]
    public class ComplexKernelProduct2DTest {
        [TestMethod]
        public void ExecuteTest() {
            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 2, 4, 10, 20, 32, 34 }) {
                    foreach (int outchannels in new int[] { 2, 4, 10, 20, 32, 34 }) {
                        foreach (int kheight in new int[] { 1, 3, 5 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                    foreach (int inheight in new int[] { kheight, kheight * 2, 8, 9, 13, 17, 25 }) {
                                        int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                        float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                        float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                        System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                                            .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

                                        System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                                            .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

                                        ComplexMap2D x = new(inchannels / 2, inwidth, inheight, batch, xcval);
                                        ComplexMap2D y = new(outchannels / 2, outwidth, outheight, batch, ycval);

                                        ComplexFilter2D gw = Reference(x, y, kwidth, kheight);

                                        OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                        OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);

                                        OverflowCheckedTensor gw_tensor = new(Shape.Kernel2D(inchannels, outchannels / 2, kwidth, kheight));

                                        ComplexKernelProduct2D ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, transpose: false, batch);

                                        ope.Execute(x_tensor, y_tensor, gw_tensor);

                                        float[] gw_expect = gw.ToArray();
                                        float[] gw_actual = gw_tensor.State.Value;

                                        CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                        CollectionAssert.AreEqual(yval, y_tensor.State.Value);

                                        AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

                                        Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

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
        public void LargeMapTest() {
            Random random = new(1234);

            float max_err = 0;

            int batch = 3;
            int inchannels = 98, outchannels = 100;
            int kwidth = 5, kheight = 3;
            int inwidth = 250, inheight = 196;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

            ComplexMap2D x = new(inchannels / 2, inwidth, inheight, batch, xcval);
            ComplexMap2D y = new(outchannels / 2, outwidth, outheight, batch, ycval);

            ComplexFilter2D gw = Reference(x, y, kwidth, kheight);

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel2D(inchannels, outchannels / 2, kwidth, kheight));

            ComplexKernelProduct2D ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, transpose: false, batch);

            ope.Execute(x_tensor, y_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(yval, y_tensor.State.Value);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inheight = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel2D(inchannels, outchannels / 2, ksize, ksize));

            ComplexKernelProduct2D ope = new(inwidth, inheight, inchannels, outchannels, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/complex_kernelproduct_2d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static ComplexFilter2D Reference(ComplexMap2D x, ComplexMap2D gy, int kwidth, int kheight) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = gy.Width, outh = gy.Height;

            if (outw != inw - kwidth + 1 || outh != inh - kheight + 1) {
                throw new ArgumentException("mismatch shape");
            }

            ComplexFilter2D w = new(inchannels, outchannels, kwidth, kheight);

            Func<System.Numerics.Complex, System.Numerics.Complex, System.Numerics.Complex> mul_grad = (z1, z2) => {
                return new System.Numerics.Complex(z1.Real * z2.Real + z1.Imaginary * z2.Imaginary, z1.Imaginary * z2.Real - z1.Real * z2.Imaginary);
            };

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ix, iy = ky, ox, oy = 0; oy < outh; iy++, oy++) {
                            for (ix = kx, ox = 0; ox < outw; ix++, ox++) {
                                for (int inch, outch = 0; outch < outchannels; outch++) {
                                    for (inch = 0; inch < inchannels; inch++) {
                                        w[inch, outch, kx, ky] += mul_grad(gy[outch, ox, oy, th], x[inch, ix, iy, th]);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 6, outchannels = 8, kwidth = 3, kheight = 5, inwidth = 13, inheight = 17;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, batch = 1;

            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

            ComplexMap2D x = new(inchannels / 2, inwidth, inheight, batch, xcval);
            ComplexMap2D y = new(outchannels / 2, outwidth, outheight, batch, ycval);

            ComplexFilter2D gw = Reference(x, y, kwidth, kheight);

            float[] gw_expect = {
                5.428623200e+01f,  -1.534390000e-01f,  5.461484600e+01f,  -1.537250000e-01f,  5.494346000e+01f,  -1.540110000e-01f,
                5.400109000e+01f,  -1.531530000e-01f,  5.432856000e+01f,  -1.534390000e-01f,  5.465603000e+01f,  -1.537250000e-01f,
                5.371594800e+01f,  -1.528670000e-01f,  5.404227400e+01f,  -1.531530000e-01f,  5.436860000e+01f,  -1.534390000e-01f,
                5.343080600e+01f,  -1.525810000e-01f,  5.375598800e+01f,  -1.528670000e-01f,  5.408117000e+01f,  -1.531530000e-01f,
                5.527207400e+01f,  -1.542970000e-01f,  5.560068800e+01f,  -1.545830000e-01f,  5.592930200e+01f,  -1.548690000e-01f,
                5.498350000e+01f,  -1.540110000e-01f,  5.531097000e+01f,  -1.542970000e-01f,  5.563844000e+01f,  -1.545830000e-01f,
                5.469492600e+01f,  -1.537250000e-01f,  5.502125200e+01f,  -1.540110000e-01f,  5.534757800e+01f,  -1.542970000e-01f,
                5.440635200e+01f,  -1.534390000e-01f,  5.473153400e+01f,  -1.537250000e-01f,  5.505671600e+01f,  -1.540110000e-01f,
                5.625791600e+01f,  -1.551550000e-01f,  5.658653000e+01f,  -1.554410000e-01f,  5.691514400e+01f,  -1.557270000e-01f,
                5.596591000e+01f,  -1.548690000e-01f,  5.629338000e+01f,  -1.551550000e-01f,  5.662085000e+01f,  -1.554410000e-01f,
                5.567390400e+01f,  -1.545830000e-01f,  5.600023000e+01f,  -1.548690000e-01f,  5.632655600e+01f,  -1.551550000e-01f,
                5.538189800e+01f,  -1.542970000e-01f,  5.570708000e+01f,  -1.545830000e-01f,  5.603226200e+01f,  -1.548690000e-01f,
                6.710217800e+01f,  -1.645930000e-01f,  6.743079200e+01f,  -1.648790000e-01f,  6.775940600e+01f,  -1.651650000e-01f,
                6.677242000e+01f,  -1.643070000e-01f,  6.709989000e+01f,  -1.645930000e-01f,  6.742736000e+01f,  -1.648790000e-01f,
                6.644266200e+01f,  -1.640210000e-01f,  6.676898800e+01f,  -1.643070000e-01f,  6.709531400e+01f,  -1.645930000e-01f,
                6.611290400e+01f,  -1.637350000e-01f,  6.643808600e+01f,  -1.640210000e-01f,  6.676326800e+01f,  -1.643070000e-01f,
                6.808802000e+01f,  -1.654510000e-01f,  6.841663400e+01f,  -1.657370000e-01f,  6.874524800e+01f,  -1.660230000e-01f,
                6.775483000e+01f,  -1.651650000e-01f,  6.808230000e+01f,  -1.654510000e-01f,  6.840977000e+01f,  -1.657370000e-01f,
                6.742164000e+01f,  -1.648790000e-01f,  6.774796600e+01f,  -1.651650000e-01f,  6.807429200e+01f,  -1.654510000e-01f,
                6.708845000e+01f,  -1.645930000e-01f,  6.741363200e+01f,  -1.648790000e-01f,  6.773881400e+01f,  -1.651650000e-01f,
                6.907386200e+01f,  -1.663090000e-01f,  6.940247600e+01f,  -1.665950000e-01f,  6.973109000e+01f,  -1.668810000e-01f,
                6.873724000e+01f,  -1.660230000e-01f,  6.906471000e+01f,  -1.663090000e-01f,  6.939218000e+01f,  -1.665950000e-01f,
                6.840061800e+01f,  -1.657370000e-01f,  6.872694400e+01f,  -1.660230000e-01f,  6.905327000e+01f,  -1.663090000e-01f,
                6.806399600e+01f,  -1.654510000e-01f,  6.838917800e+01f,  -1.657370000e-01f,  6.871436000e+01f,  -1.660230000e-01f,
                7.991812400e+01f,  -1.757470000e-01f,  8.024673800e+01f,  -1.760330000e-01f,  8.057535200e+01f,  -1.763190000e-01f,
                7.954375000e+01f,  -1.754610000e-01f,  7.987122000e+01f,  -1.757470000e-01f,  8.019869000e+01f,  -1.760330000e-01f,
                7.916937600e+01f,  -1.751750000e-01f,  7.949570200e+01f,  -1.754610000e-01f,  7.982202800e+01f,  -1.757470000e-01f,
                7.879500200e+01f,  -1.748890000e-01f,  7.912018400e+01f,  -1.751750000e-01f,  7.944536600e+01f,  -1.754610000e-01f,
                8.090396600e+01f,  -1.766050000e-01f,  8.123258000e+01f,  -1.768910000e-01f,  8.156119400e+01f,  -1.771770000e-01f,
                8.052616000e+01f,  -1.763190000e-01f,  8.085363000e+01f,  -1.766050000e-01f,  8.118110000e+01f,  -1.768910000e-01f,
                8.014835400e+01f,  -1.760330000e-01f,  8.047468000e+01f,  -1.763190000e-01f,  8.080100600e+01f,  -1.766050000e-01f,
                7.977054800e+01f,  -1.757470000e-01f,  8.009573000e+01f,  -1.760330000e-01f,  8.042091200e+01f,  -1.763190000e-01f,
                8.188980800e+01f,  -1.774630000e-01f,  8.221842200e+01f,  -1.777490000e-01f,  8.254703600e+01f,  -1.780350000e-01f,
                8.150857000e+01f,  -1.771770000e-01f,  8.183604000e+01f,  -1.774630000e-01f,  8.216351000e+01f,  -1.777490000e-01f,
                8.112733200e+01f,  -1.768910000e-01f,  8.145365800e+01f,  -1.771770000e-01f,  8.177998400e+01f,  -1.774630000e-01f,
                8.074609400e+01f,  -1.766050000e-01f,  8.107127600e+01f,  -1.768910000e-01f,  8.139645800e+01f,  -1.771770000e-01f,
                9.273407000e+01f,  -1.869010000e-01f,  9.306268400e+01f,  -1.871870000e-01f,  9.339129800e+01f,  -1.874730000e-01f,
                9.231508000e+01f,  -1.866150000e-01f,  9.264255000e+01f,  -1.869010000e-01f,  9.297002000e+01f,  -1.871870000e-01f,
                9.189609000e+01f,  -1.863290000e-01f,  9.222241600e+01f,  -1.866150000e-01f,  9.254874200e+01f,  -1.869010000e-01f,
                9.147710000e+01f,  -1.860430000e-01f,  9.180228200e+01f,  -1.863290000e-01f,  9.212746400e+01f,  -1.866150000e-01f,
                9.371991200e+01f,  -1.877590000e-01f,  9.404852600e+01f,  -1.880450000e-01f,  9.437714000e+01f,  -1.883310000e-01f,
                9.329749000e+01f,  -1.874730000e-01f,  9.362496000e+01f,  -1.877590000e-01f,  9.395243000e+01f,  -1.880450000e-01f,
                9.287506800e+01f,  -1.871870000e-01f,  9.320139400e+01f,  -1.874730000e-01f,  9.352772000e+01f,  -1.877590000e-01f,
                9.245264600e+01f,  -1.869010000e-01f,  9.277782800e+01f,  -1.871870000e-01f,  9.310301000e+01f,  -1.874730000e-01f,
                9.470575400e+01f,  -1.886170000e-01f,  9.503436800e+01f,  -1.889030000e-01f,  9.536298200e+01f,  -1.891890000e-01f,
                9.427990000e+01f,  -1.883310000e-01f,  9.460737000e+01f,  -1.886170000e-01f,  9.493484000e+01f,  -1.889030000e-01f,
                9.385404600e+01f,  -1.880450000e-01f,  9.418037200e+01f,  -1.883310000e-01f,  9.450669800e+01f,  -1.886170000e-01f,
                9.342819200e+01f,  -1.877590000e-01f,  9.375337400e+01f,  -1.880450000e-01f,  9.407855600e+01f,  -1.883310000e-01f,
                1.055500160e+02f,  -1.980550000e-01f,  1.058786300e+02f,  -1.983410000e-01f,  1.062072440e+02f,  -1.986270000e-01f,
                1.050864100e+02f,  -1.977690000e-01f,  1.054138800e+02f,  -1.980550000e-01f,  1.057413500e+02f,  -1.983410000e-01f,
                1.046228040e+02f,  -1.974830000e-01f,  1.049491300e+02f,  -1.977690000e-01f,  1.052754560e+02f,  -1.980550000e-01f,
                1.041591980e+02f,  -1.971970000e-01f,  1.044843800e+02f,  -1.974830000e-01f,  1.048095620e+02f,  -1.977690000e-01f,
                1.065358580e+02f,  -1.989130000e-01f,  1.068644720e+02f,  -1.991990000e-01f,  1.071930860e+02f,  -1.994850000e-01f,
                1.060688200e+02f,  -1.986270000e-01f,  1.063962900e+02f,  -1.989130000e-01f,  1.067237600e+02f,  -1.991990000e-01f,
                1.056017820e+02f,  -1.983410000e-01f,  1.059281080e+02f,  -1.986270000e-01f,  1.062544340e+02f,  -1.989130000e-01f,
                1.051347440e+02f,  -1.980550000e-01f,  1.054599260e+02f,  -1.983410000e-01f,  1.057851080e+02f,  -1.986270000e-01f,
                1.075217000e+02f,  -1.997710000e-01f,  1.078503140e+02f,  -2.000570000e-01f,  1.081789280e+02f,  -2.003430000e-01f,
                1.070512300e+02f,  -1.994850000e-01f,  1.073787000e+02f,  -1.997710000e-01f,  1.077061700e+02f,  -2.000570000e-01f,
                1.065807600e+02f,  -1.991990000e-01f,  1.069070860e+02f,  -1.994850000e-01f,  1.072334120e+02f,  -1.997710000e-01f,
                1.061102900e+02f,  -1.989130000e-01f,  1.064354720e+02f,  -1.991990000e-01f,  1.067606540e+02f,  -1.994850000e-01f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
        }
    }
}
