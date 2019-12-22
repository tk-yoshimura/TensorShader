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

                                ComplexMap1D x = Reference(y, w, inwidth, kwidth);

                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);
                                OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 2, kwidth), wval);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch));

                                ComplexDeconvolution1D ope = new ComplexDeconvolution1D(inwidth, outchannels, inchannels, kwidth, gradmode: false, batch);

                                ope.Execute(y_tensor, w_tensor, x_tensor);

                                float[] x_expect = x.ToArray();
                                float[] x_actual = x_tensor.State;

                                CollectionAssert.AreEqual(yval, y_tensor.State);
                                CollectionAssert.AreEqual(wval, w_tensor.State);

                                AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

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

                                    float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * inchannels * outchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);
                                    OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 2, kwidth), wval);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch));

                                    ComplexDeconvolution1D ope = new ComplexDeconvolution1D(inwidth, outchannels, inchannels, kwidth, gradmode, batch);

                                    ope.Execute(y_tensor, w_tensor, x_tensor);

                                    CollectionAssert.AreEqual(yval, y_tensor.State);
                                    CollectionAssert.AreEqual(wval, w_tensor.State);

                                    x_tensor.CheckOverflow();

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

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 2, ksize));

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));

            ComplexDeconvolution1D ope = new ComplexDeconvolution1D(inwidth, outchannels, inchannels, ksize);

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

        public static ComplexMap1D Reference(ComplexMap1D y, ComplexFilter1D w, int inw, int kwidth) {
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
                                x[inch, kx + ox, th] += v * w[inch, outch, kx];
                            }
                        }
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 6, outchannels = 8, kwidth = 3, inwidth = 13, batch = 3;
            int outwidth = inwidth - kwidth + 1;

            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

            ComplexMap1D y = new ComplexMap1D(outchannels / 2, outwidth, batch, ycval);
            ComplexFilter1D w = new ComplexFilter1D(inchannels / 2, outchannels / 2, kwidth, wcval);

            ComplexMap1D x = Reference(y, w, inwidth, kwidth);

            float[] x_expect = {
                -2.320000000e-04f,  1.604000000e-03f,  -2.240000000e-04f,  1.548000000e-03f,  -2.160000000e-04f,  1.492000000e-03f, 
                -3.360000000e-04f,  6.472000000e-03f,  -3.200000000e-04f,  6.232000000e-03f,  -3.040000000e-04f,  5.992000000e-03f, 
                -3.120000000e-04f,  1.306800000e-02f,  -2.880000000e-04f,  1.251600000e-02f,  -2.640000000e-04f,  1.196400000e-02f, 
                -2.160000000e-04f,  2.026800000e-02f,  -1.920000000e-04f,  1.933200000e-02f,  -1.680000000e-04f,  1.839600000e-02f, 
                -1.200000000e-04f,  2.746800000e-02f,  -9.600000000e-05f,  2.614800000e-02f,  -7.200000000e-05f,  2.482800000e-02f, 
                -2.400000000e-05f,  3.466800000e-02f,  0.000000000e+00f,  3.296400000e-02f,  2.400000000e-05f,  3.126000000e-02f, 
                7.200000000e-05f,  4.186800000e-02f,  9.600000000e-05f,  3.978000000e-02f,  1.200000000e-04f,  3.769200000e-02f, 
                1.680000000e-04f,  4.906800000e-02f,  1.920000000e-04f,  4.659600000e-02f,  2.160000000e-04f,  4.412400000e-02f, 
                2.640000000e-04f,  5.626800000e-02f,  2.880000000e-04f,  5.341200000e-02f,  3.120000000e-04f,  5.055600000e-02f, 
                3.600000000e-04f,  6.346800000e-02f,  3.840000000e-04f,  6.022800000e-02f,  4.080000000e-04f,  5.698800000e-02f, 
                4.560000000e-04f,  7.066800000e-02f,  4.800000000e-04f,  6.704400000e-02f,  5.040000000e-04f,  6.342000000e-02f, 
                4.320000000e-04f,  3.296800000e-02f,  4.480000000e-04f,  3.042400000e-02f,  4.640000000e-04f,  2.788000000e-02f, 
                2.800000000e-04f,  8.900000000e-03f,  2.880000000e-04f,  7.564000000e-03f,  2.960000000e-04f,  6.228000000e-03f, 
                1.200000000e-04f,  4.490000000e-02f,  1.280000000e-04f,  4.343600000e-02f,  1.360000000e-04f,  4.197200000e-02f, 
                3.680000000e-04f,  7.616800000e-02f,  3.840000000e-04f,  7.311200000e-02f,  4.000000000e-04f,  7.005600000e-02f, 
                7.440000000e-04f,  9.226800000e-02f,  7.680000000e-04f,  8.749200000e-02f,  7.920000000e-04f,  8.271600000e-02f, 
                8.400000000e-04f,  9.946800000e-02f,  8.640000000e-04f,  9.430800000e-02f,  8.880000000e-04f,  8.914800000e-02f, 
                9.360000000e-04f,  1.066680000e-01f,  9.600000000e-04f,  1.011240000e-01f,  9.840000000e-04f,  9.558000000e-02f, 
                1.032000000e-03f,  1.138680000e-01f,  1.056000000e-03f,  1.079400000e-01f,  1.080000000e-03f,  1.020120000e-01f, 
                1.128000000e-03f,  1.210680000e-01f,  1.152000000e-03f,  1.147560000e-01f,  1.176000000e-03f,  1.084440000e-01f, 
                1.224000000e-03f,  1.282680000e-01f,  1.248000000e-03f,  1.215720000e-01f,  1.272000000e-03f,  1.148760000e-01f, 
                1.320000000e-03f,  1.354680000e-01f,  1.344000000e-03f,  1.283880000e-01f,  1.368000000e-03f,  1.213080000e-01f, 
                1.416000000e-03f,  1.426680000e-01f,  1.440000000e-03f,  1.352040000e-01f,  1.464000000e-03f,  1.277400000e-01f, 
                1.512000000e-03f,  1.498680000e-01f,  1.536000000e-03f,  1.420200000e-01f,  1.560000000e-03f,  1.341720000e-01f, 
                1.136000000e-03f,  6.887200000e-02f,  1.152000000e-03f,  6.351200000e-02f,  1.168000000e-03f,  5.815200000e-02f, 
                6.320000000e-04f,  1.840400000e-02f,  6.400000000e-04f,  1.566000000e-02f,  6.480000000e-04f,  1.291600000e-02f, 
                4.720000000e-04f,  8.819600000e-02f,  4.800000000e-04f,  8.532400000e-02f,  4.880000000e-04f,  8.245200000e-02f, 
                1.072000000e-03f,  1.458640000e-01f,  1.088000000e-03f,  1.399920000e-01f,  1.104000000e-03f,  1.341200000e-01f, 
                1.800000000e-03f,  1.714680000e-01f,  1.824000000e-03f,  1.624680000e-01f,  1.848000000e-03f,  1.534680000e-01f, 
                1.896000000e-03f,  1.786680000e-01f,  1.920000000e-03f,  1.692840000e-01f,  1.944000000e-03f,  1.599000000e-01f, 
                1.992000000e-03f,  1.858680000e-01f,  2.016000000e-03f,  1.761000000e-01f,  2.040000000e-03f,  1.663320000e-01f, 
                2.088000000e-03f,  1.930680000e-01f,  2.112000000e-03f,  1.829160000e-01f,  2.136000000e-03f,  1.727640000e-01f, 
                2.184000000e-03f,  2.002680000e-01f,  2.208000000e-03f,  1.897320000e-01f,  2.232000000e-03f,  1.791960000e-01f, 
                2.280000000e-03f,  2.074680000e-01f,  2.304000000e-03f,  1.965480000e-01f,  2.328000000e-03f,  1.856280000e-01f, 
                2.376000000e-03f,  2.146680000e-01f,  2.400000000e-03f,  2.033640000e-01f,  2.424000000e-03f,  1.920600000e-01f, 
                2.472000000e-03f,  2.218680000e-01f,  2.496000000e-03f,  2.101800000e-01f,  2.520000000e-03f,  1.984920000e-01f, 
                2.568000000e-03f,  2.290680000e-01f,  2.592000000e-03f,  2.169960000e-01f,  2.616000000e-03f,  2.049240000e-01f, 
                1.840000000e-03f,  1.047760000e-01f,  1.856000000e-03f,  9.660000000e-02f,  1.872000000e-03f,  8.842400000e-02f, 
                9.840000000e-04f,  2.790800000e-02f,  9.920000000e-04f,  2.375600000e-02f,  1.000000000e-03f,  1.960400000e-02f, 
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");
        }
    }
}
