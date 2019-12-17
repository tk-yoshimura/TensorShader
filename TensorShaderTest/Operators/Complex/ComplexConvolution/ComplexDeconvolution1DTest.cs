using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ComplexConvolution;

namespace TensorShaderTest.Operators.Complex {
    [TestClass]
    public class ComplexDeconvolution1DTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int inchannels in new int[] { 2, 4, 10, 20 }) {
                    foreach (int outchannels in new int[] { 6, 14 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int stride in new int[] { 1, 2, 3 }) {
                                foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                    int outwidth = inwidth - kwidth + 1;

                                    float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * inchannels * outchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                                        .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

                                    System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                                        .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

                                    ComplexMap1D y = new ComplexMap1D(outchannels / 2, outwidth, batch, ycval);
                                    ComplexFilter1D w = new ComplexFilter1D(inchannels / 2, outchannels / 2, kwidth, wcval);

                                    ComplexMap1D x = Reference(y, w, inwidth, kwidth, stride);

                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);
                                    OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 2, kwidth), wval);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch));

                                    ComplexDeconvolution1D ope = new ComplexDeconvolution1D(inwidth, outchannels, inchannels, kwidth, stride, gradmode: false, batch);

                                    ope.Execute(y_tensor, w_tensor, x_tensor);

                                    float[] x_expect = x.ToArray();
                                    float[] x_actual = x_tensor.State;

                                    CollectionAssert.AreEqual(yval, y_tensor.State);
                                    CollectionAssert.AreEqual(wval, w_tensor.State);

                                    AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");

                                    Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");
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
            foreach (bool gradmode in new bool[] { false, true }) {
                foreach (int batch in new int[] { 1, 2, 3 }) {
                    foreach (int inchannels in new int[] { 2, 4, 10, 20 }) {
                        foreach (int outchannels in new int[] { 6, 14 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int stride in new int[] { 1, 2, 3 }) {
                                    foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                        int outwidth = inwidth - kwidth + 1;

                                        float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                        float[] wval = (new float[kwidth * inchannels * outchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);
                                        OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 2, kwidth), wval);

                                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch));

                                        ComplexDeconvolution1D ope = new ComplexDeconvolution1D(inwidth, outchannels, inchannels, kwidth, stride, gradmode, batch);

                                        ope.Execute(y_tensor, w_tensor, x_tensor);

                                        CollectionAssert.AreEqual(yval, y_tensor.State);
                                        CollectionAssert.AreEqual(wval, w_tensor.State);

                                        x_tensor.CheckOverflow();

                                        Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch},{gradmode}");
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
            int inwidth = 512, inchannels = 32, outchannels = 32, ksize = 3, stride = 2;
            int outwidth = (inwidth - ksize) / stride + 1;

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 2, ksize));

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));

            ComplexDeconvolution1D ope = new ComplexDeconvolution1D(inwidth, outchannels, inchannels, ksize, stride);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);
            ope.Execute(y_tensor, w_tensor, x_tensor);
            ope.Execute(y_tensor, w_tensor, x_tensor);
            ope.Execute(y_tensor, w_tensor, x_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static ComplexMap1D Reference(ComplexMap1D y, ComplexFilter1D w, int inw, int kwidth, int stride) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int outw = inw - kwidth + 1;

            if (y.Width != outw) {
                throw new ArgumentException("mismatch shape");
            }

            ComplexMap1D x = new ComplexMap1D(inchannels, inw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        for (int outch = 0; outch < outchannels; outch++) {
                            System.Numerics.Complex v = y[outch, ox, th];

                            for (int inch = 0; inch < inchannels; inch++) {
                                x[inch, kx + ox * stride, th] += v * w[inch, outch, kx];
                            }
                        }
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 6, outchannels = 8, kwidth = 3, stride = 2, inwidth = 13, batch = 3;
            int outwidth = inwidth - kwidth + 1;

            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

            ComplexMap1D y = new ComplexMap1D(outchannels / 2, outwidth, batch, ycval);
            ComplexFilter1D w = new ComplexFilter1D(inchannels / 2, outchannels / 2, kwidth, wcval);

            ComplexMap1D x = Reference(y, w, inwidth, kwidth, stride);

            float[] x_expect = {
                -2.320000000e-04f,  1.604000000e-03f,  -2.240000000e-04f,  1.548000000e-03f,  -2.160000000e-04f,  1.492000000e-03f,
                -1.360000000e-04f,  9.320000000e-04f,  -1.280000000e-04f,  8.760000000e-04f,  -1.200000000e-04f,  8.200000000e-04f,
                -2.400000000e-04f,  5.800000000e-03f,  -2.240000000e-04f,  5.560000000e-03f,  -2.080000000e-04f,  5.320000000e-03f,
                -1.040000000e-04f,  3.332000000e-03f,  -9.600000000e-05f,  3.148000000e-03f,  -8.800000000e-05f,  2.964000000e-03f,
                -1.760000000e-04f,  1.060000000e-02f,  -1.600000000e-04f,  1.010400000e-02f,  -1.440000000e-04f,  9.608000000e-03f,
                -7.200000000e-05f,  5.732000000e-03f,  -6.400000000e-05f,  5.420000000e-03f,  -5.600000000e-05f,  5.108000000e-03f,
                -1.120000000e-04f,  1.540000000e-02f,  -9.600000000e-05f,  1.464800000e-02f,  -8.000000000e-05f,  1.389600000e-02f,
                -4.000000000e-05f,  8.132000000e-03f,  -3.200000000e-05f,  7.692000000e-03f,  -2.400000000e-05f,  7.252000000e-03f,
                -4.800000000e-05f,  2.020000000e-02f,  -3.200000000e-05f,  1.919200000e-02f,  -1.600000000e-05f,  1.818400000e-02f,
                -8.000000000e-06f,  1.053200000e-02f,  0.000000000e+00f,  9.964000000e-03f,  8.000000000e-06f,  9.396000000e-03f,
                1.600000000e-05f,  2.500000000e-02f,  3.200000000e-05f,  2.373600000e-02f,  4.800000000e-05f,  2.247200000e-02f,
                2.400000000e-05f,  1.293200000e-02f,  3.200000000e-05f,  1.223600000e-02f,  4.000000000e-05f,  1.154000000e-02f,
                1.200000000e-04f,  4.580000000e-03f,  1.280000000e-04f,  3.884000000e-03f,  1.360000000e-04f,  3.188000000e-03f,
                -4.000000000e-05f,  2.522000000e-02f,  -3.200000000e-05f,  2.439600000e-02f,  -2.400000000e-05f,  2.357200000e-02f,
                5.600000000e-05f,  1.533200000e-02f,  6.400000000e-05f,  1.450800000e-02f,  7.200000000e-05f,  1.368400000e-02f,
                1.440000000e-04f,  3.460000000e-02f,  1.600000000e-04f,  3.282400000e-02f,  1.760000000e-04f,  3.104800000e-02f,
                8.800000000e-05f,  1.773200000e-02f,  9.600000000e-05f,  1.678000000e-02f,  1.040000000e-04f,  1.582800000e-02f,
                2.080000000e-04f,  3.940000000e-02f,  2.240000000e-04f,  3.736800000e-02f,  2.400000000e-04f,  3.533600000e-02f,
                1.200000000e-04f,  2.013200000e-02f,  1.280000000e-04f,  1.905200000e-02f,  1.360000000e-04f,  1.797200000e-02f,
                2.720000000e-04f,  4.420000000e-02f,  2.880000000e-04f,  4.191200000e-02f,  3.040000000e-04f,  3.962400000e-02f,
                1.520000000e-04f,  2.253200000e-02f,  1.600000000e-04f,  2.132400000e-02f,  1.680000000e-04f,  2.011600000e-02f,
                3.360000000e-04f,  4.900000000e-02f,  3.520000000e-04f,  4.645600000e-02f,  3.680000000e-04f,  4.391200000e-02f,
                1.840000000e-04f,  2.493200000e-02f,  1.920000000e-04f,  2.359600000e-02f,  2.000000000e-04f,  2.226000000e-02f,
                4.000000000e-04f,  5.380000000e-02f,  4.160000000e-04f,  5.100000000e-02f,  4.320000000e-04f,  4.820000000e-02f,
                2.160000000e-04f,  2.733200000e-02f,  2.240000000e-04f,  2.586800000e-02f,  2.320000000e-04f,  2.440400000e-02f,
                3.120000000e-04f,  9.764000000e-03f,  3.200000000e-04f,  8.300000000e-03f,  3.280000000e-04f,  6.836000000e-03f,
                1.520000000e-04f,  4.883600000e-02f,  1.600000000e-04f,  4.724400000e-02f,  1.680000000e-04f,  4.565200000e-02f,
                2.480000000e-04f,  2.973200000e-02f,  2.560000000e-04f,  2.814000000e-02f,  2.640000000e-04f,  2.654800000e-02f,
                5.280000000e-04f,  6.340000000e-02f,  5.440000000e-04f,  6.008800000e-02f,  5.600000000e-04f,  5.677600000e-02f,
                2.800000000e-04f,  3.213200000e-02f,  2.880000000e-04f,  3.041200000e-02f,  2.960000000e-04f,  2.869200000e-02f,
                5.920000000e-04f,  6.820000000e-02f,  6.080000000e-04f,  6.463200000e-02f,  6.240000000e-04f,  6.106400000e-02f,
                3.120000000e-04f,  3.453200000e-02f,  3.200000000e-04f,  3.268400000e-02f,  3.280000000e-04f,  3.083600000e-02f,
                6.560000000e-04f,  7.300000000e-02f,  6.720000000e-04f,  6.917600000e-02f,  6.880000000e-04f,  6.535200000e-02f,
                3.440000000e-04f,  3.693200000e-02f,  3.520000000e-04f,  3.495600000e-02f,  3.600000000e-04f,  3.298000000e-02f,
                7.200000000e-04f,  7.780000000e-02f,  7.360000000e-04f,  7.372000000e-02f,  7.520000000e-04f,  6.964000000e-02f,
                3.760000000e-04f,  3.933200000e-02f,  3.840000000e-04f,  3.722800000e-02f,  3.920000000e-04f,  3.512400000e-02f,
                7.840000000e-04f,  8.260000000e-02f,  8.000000000e-04f,  7.826400000e-02f,  8.160000000e-04f,  7.392800000e-02f,
                4.080000000e-04f,  4.173200000e-02f,  4.160000000e-04f,  3.950000000e-02f,  4.240000000e-04f,  3.726800000e-02f,
                5.040000000e-04f,  1.494800000e-02f,  5.120000000e-04f,  1.271600000e-02f,  5.200000000e-04f,  1.048400000e-02f,
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");
        }
    }
}
