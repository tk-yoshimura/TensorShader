using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ComplexConvolution;

namespace TensorShaderTest.Operators.Complex {
    [TestClass]
    public class ComplexConvolution1DTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int inchannels in new int[] { 2, 4, 10, 20 }) {
                    foreach (int outchannels in new int[] { 6, 14 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                int outwidth = inwidth - kwidth + 1;

                                float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] wval = (new float[kwidth * inchannels * outchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                                    .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

                                System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                                    .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

                                ComplexMap1D x = new ComplexMap1D(inchannels / 2, inwidth, batch, xcval);
                                ComplexFilter1D w = new ComplexFilter1D(inchannels / 2, outchannels / 2, kwidth, wcval);

                                ComplexMap1D y = Reference(x, w, kwidth);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
                                OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 2, kwidth), wval);

                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch));

                                ComplexConvolution1D ope = new ComplexConvolution1D(inwidth, inchannels, outchannels, kwidth, gradmode: false, batch);

                                ope.Execute(x_tensor, w_tensor, y_tensor);

                                float[] y_expect = y.ToArray();
                                float[] y_actual = y_tensor.State;

                                CollectionAssert.AreEqual(xval, x_tensor.State);
                                CollectionAssert.AreEqual(wval, w_tensor.State);

                                AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void OverflowTest() {
            foreach (bool gradmode in new bool[] { false, true }) {
                foreach (int batch in new int[] { 1, 2, 3 }) {
                    foreach (int inchannels in new int[] { 2, 4, 10, 20 }) {
                        foreach (int outchannels in new int[] { 6, 14 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                    int outwidth = inwidth - kwidth + 1;

                                    float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * inchannels * outchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
                                    OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 2, kwidth), wval);

                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch));

                                    ComplexConvolution1D ope = new ComplexConvolution1D(inwidth, inchannels, outchannels, kwidth, gradmode, batch);

                                    ope.Execute(x_tensor, w_tensor, y_tensor);

                                    CollectionAssert.AreEqual(xval, x_tensor.State);
                                    CollectionAssert.AreEqual(wval, w_tensor.State);

                                    y_tensor.CheckOverflow();

                                    Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch},{gradmode}");
                                }
                            }
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 2, ksize));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth));

            ComplexConvolution1D ope = new ComplexConvolution1D(inwidth, inchannels, outchannels, ksize);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static ComplexMap1D Reference(ComplexMap1D x, ComplexFilter1D w, int kwidth) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, outw = inw - kwidth + 1;

            ComplexMap1D y = new ComplexMap1D(outchannels, outw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        for (int outch = 0; outch < outchannels; outch++) {
                            System.Numerics.Complex sum = y[outch, ox, th];

                            for (int inch = 0; inch < inchannels; inch++) {
                                sum += x[inch, kx + ox, th] * w[inch, outch, kx];
                            }

                            y[outch, ox, th] = sum;
                        }
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 6, outchannels = 8, kwidth = 3, inwidth = 13, batch = 3;
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[batch * inwidth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

            ComplexMap1D x = new ComplexMap1D(inchannels / 2, inwidth, batch, xcval);
            ComplexFilter1D w = new ComplexFilter1D(inchannels / 2, outchannels / 2, kwidth, wcval);

            ComplexMap1D y = Reference(x, w, kwidth);

            float[] y_expect = {
                -3.240000000e-04f,  5.037000000e-03f,  -2.700000000e-04f,  4.119000000e-03f,  -2.160000000e-04f,  3.201000000e-03f, 
                -1.620000000e-04f,  2.283000000e-03f,  -2.700000000e-04f,  9.843000000e-03f,  -2.160000000e-04f,  8.277000000e-03f, 
                -1.620000000e-04f,  6.711000000e-03f,  -1.080000000e-04f,  5.145000000e-03f,  -2.160000000e-04f,  1.464900000e-02f, 
                -1.620000000e-04f,  1.243500000e-02f,  -1.080000000e-04f,  1.022100000e-02f,  -5.400000000e-05f,  8.007000000e-03f, 
                -1.620000000e-04f,  1.945500000e-02f,  -1.080000000e-04f,  1.659300000e-02f,  -5.400000000e-05f,  1.373100000e-02f, 
                0.000000000e+00f,  1.086900000e-02f,  -1.080000000e-04f,  2.426100000e-02f,  -5.400000000e-05f,  2.075100000e-02f, 
                1.734723476e-18f,  1.724100000e-02f,  5.400000000e-05f,  1.373100000e-02f,  -5.400000000e-05f,  2.906700000e-02f, 
                0.000000000e+00f,  2.490900000e-02f,  5.400000000e-05f,  2.075100000e-02f,  1.080000000e-04f,  1.659300000e-02f, 
                0.000000000e+00f,  3.387300000e-02f,  5.400000000e-05f,  2.906700000e-02f,  1.080000000e-04f,  2.426100000e-02f, 
                1.620000000e-04f,  1.945500000e-02f,  5.400000000e-05f,  3.867900000e-02f,  1.080000000e-04f,  3.322500000e-02f, 
                1.620000000e-04f,  2.777100000e-02f,  2.160000000e-04f,  2.231700000e-02f,  1.080000000e-04f,  4.348500000e-02f, 
                1.620000000e-04f,  3.738300000e-02f,  2.160000000e-04f,  3.128100000e-02f,  2.700000000e-04f,  2.517900000e-02f, 
                1.620000000e-04f,  4.829100000e-02f,  2.160000000e-04f,  4.154100000e-02f,  2.700000000e-04f,  3.479100000e-02f, 
                3.240000000e-04f,  2.804100000e-02f,  2.160000000e-04f,  5.309700000e-02f,  2.700000000e-04f,  4.569900000e-02f, 
                3.240000000e-04f,  3.830100000e-02f,  3.780000000e-04f,  3.090300000e-02f,  3.780000000e-04f,  6.751500000e-02f, 
                4.320000000e-04f,  5.817300000e-02f,  4.860000000e-04f,  4.883100000e-02f,  5.400000000e-04f,  3.948900000e-02f, 
                4.320000000e-04f,  7.232100000e-02f,  4.860000000e-04f,  6.233100000e-02f,  5.400000000e-04f,  5.234100000e-02f, 
                5.940000000e-04f,  4.235100000e-02f,  4.860000000e-04f,  7.712700000e-02f,  5.400000000e-04f,  6.648900000e-02f, 
                5.940000000e-04f,  5.585100000e-02f,  6.480000000e-04f,  4.521300000e-02f,  5.400000000e-04f,  8.193300000e-02f, 
                5.940000000e-04f,  7.064700000e-02f,  6.480000000e-04f,  5.936100000e-02f,  7.020000000e-04f,  4.807500000e-02f, 
                5.940000000e-04f,  8.673900000e-02f,  6.480000000e-04f,  7.480500000e-02f,  7.020000000e-04f,  6.287100000e-02f, 
                7.560000000e-04f,  5.093700000e-02f,  6.480000000e-04f,  9.154500000e-02f,  7.020000000e-04f,  7.896300000e-02f, 
                7.560000000e-04f,  6.638100000e-02f,  8.100000000e-04f,  5.379900000e-02f,  7.020000000e-04f,  9.635100000e-02f, 
                7.560000000e-04f,  8.312100000e-02f,  8.100000000e-04f,  6.989100000e-02f,  8.640000000e-04f,  5.666100000e-02f, 
                7.560000000e-04f,  1.011570000e-01f,  8.100000000e-04f,  8.727900000e-02f,  8.640000000e-04f,  7.340100000e-02f, 
                9.180000000e-04f,  5.952300000e-02f,  8.100000000e-04f,  1.059630000e-01f,  8.640000000e-04f,  9.143700000e-02f, 
                9.180000000e-04f,  7.691100000e-02f,  9.720000000e-04f,  6.238500000e-02f,  8.640000000e-04f,  1.107690000e-01f, 
                9.180000000e-04f,  9.559500000e-02f,  9.720000000e-04f,  8.042100000e-02f,  1.026000000e-03f,  6.524700000e-02f, 
                9.180000000e-04f,  1.155750000e-01f,  9.720000000e-04f,  9.975300000e-02f,  1.026000000e-03f,  8.393100000e-02f, 
                1.080000000e-03f,  6.810900000e-02f,  1.080000000e-03f,  1.299930000e-01f,  1.134000000e-03f,  1.122270000e-01f, 
                1.188000000e-03f,  9.446100000e-02f,  1.242000000e-03f,  7.669500000e-02f,  1.134000000e-03f,  1.347990000e-01f, 
                1.188000000e-03f,  1.163850000e-01f,  1.242000000e-03f,  9.797100000e-02f,  1.296000000e-03f,  7.955700000e-02f, 
                1.188000000e-03f,  1.396050000e-01f,  1.242000000e-03f,  1.205430000e-01f,  1.296000000e-03f,  1.014810000e-01f, 
                1.350000000e-03f,  8.241900000e-02f,  1.242000000e-03f,  1.444110000e-01f,  1.296000000e-03f,  1.247010000e-01f, 
                1.350000000e-03f,  1.049910000e-01f,  1.404000000e-03f,  8.528100000e-02f,  1.296000000e-03f,  1.492170000e-01f, 
                1.350000000e-03f,  1.288590000e-01f,  1.404000000e-03f,  1.085010000e-01f,  1.458000000e-03f,  8.814300000e-02f, 
                1.350000000e-03f,  1.540230000e-01f,  1.404000000e-03f,  1.330170000e-01f,  1.458000000e-03f,  1.120110000e-01f, 
                1.512000000e-03f,  9.100500000e-02f,  1.404000000e-03f,  1.588290000e-01f,  1.458000000e-03f,  1.371750000e-01f, 
                1.512000000e-03f,  1.155210000e-01f,  1.566000000e-03f,  9.386700000e-02f,  1.458000000e-03f,  1.636350000e-01f, 
                1.512000000e-03f,  1.413330000e-01f,  1.566000000e-03f,  1.190310000e-01f,  1.620000000e-03f,  9.672900000e-02f, 
                1.512000000e-03f,  1.684410000e-01f,  1.566000000e-03f,  1.454910000e-01f,  1.620000000e-03f,  1.225410000e-01f, 
                1.674000000e-03f,  9.959100000e-02f,  1.566000000e-03f,  1.732470000e-01f,  1.620000000e-03f,  1.496490000e-01f, 
                1.674000000e-03f,  1.260510000e-01f,  1.728000000e-03f,  1.024530000e-01f,  1.620000000e-03f,  1.780530000e-01f, 
                1.674000000e-03f,  1.538070000e-01f,  1.728000000e-03f,  1.295610000e-01f,  1.782000000e-03f,  1.053150000e-01f, 
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");
        }
    }
}
