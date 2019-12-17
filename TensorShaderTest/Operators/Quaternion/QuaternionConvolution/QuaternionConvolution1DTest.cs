using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.QuaternionConvolution;

namespace TensorShaderTest.Operators.Quaternion {
    [TestClass]
    public class QuaternionConvolution1DTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int inchannels in new int[] { 4, 8, 12 }) {
                    foreach (int outchannels in new int[] { 4, 8, 12 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int stride in new int[] { 1, 2, 3 }) {
                                foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                    int outwidth = inwidth - kwidth + 1;

                                    float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * inchannels * outchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                                        .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

                                    Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                                        .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                    QuaternionMap1D x = new QuaternionMap1D(inchannels / 4, inwidth, batch, xcval);
                                    QuaternionFilter1D w = new QuaternionFilter1D(inchannels / 4, outchannels / 4, kwidth, wcval);

                                    QuaternionMap1D y = Reference(x, w, kwidth, stride);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
                                    OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, kwidth), wval);

                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch));

                                    QuaternionConvolution1D ope = new QuaternionConvolution1D(inwidth, inchannels, outchannels, kwidth, stride, gradmode: false, batch);

                                    ope.Execute(x_tensor, w_tensor, y_tensor);

                                    float[] y_expect = y.ToArray();
                                    float[] y_actual = y_tensor.State;

                                    CollectionAssert.AreEqual(xval, x_tensor.State);
                                    CollectionAssert.AreEqual(wval, w_tensor.State);

                                    AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");

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
                    foreach (int inchannels in new int[] { 4, 8, 12 }) {
                        foreach (int outchannels in new int[] { 4, 8, 12 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int stride in new int[] { 1, 2, 3 }) {
                                    foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                        int outwidth = inwidth - kwidth + 1;

                                        float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                        float[] wval = (new float[kwidth * inchannels * outchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                        Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                                            .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

                                        Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                                            .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                        QuaternionMap1D x = new QuaternionMap1D(inchannels / 4, inwidth, batch, xcval);
                                        QuaternionFilter1D w = new QuaternionFilter1D(inchannels / 4, outchannels / 4, kwidth, wcval);

                                        QuaternionMap1D y = Reference(x, w, kwidth, stride);

                                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
                                        OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, kwidth), wval);

                                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch));

                                        QuaternionConvolution1D ope = new QuaternionConvolution1D(inwidth, inchannels, outchannels, kwidth, stride, gradmode, batch);

                                        ope.Execute(x_tensor, w_tensor, y_tensor);

                                        CollectionAssert.AreEqual(xval, x_tensor.State);
                                        CollectionAssert.AreEqual(wval, w_tensor.State);

                                        y_tensor.CheckOverflow();

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

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, ksize));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth));

            QuaternionConvolution1D ope = new QuaternionConvolution1D(inwidth, inchannels, outchannels, ksize, stride);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static QuaternionMap1D Reference(QuaternionMap1D x, QuaternionFilter1D w, int kwidth, int stride) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, outw = inw - kwidth + 1;

            QuaternionMap1D y = new QuaternionMap1D(outchannels, outw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        for (int outch = 0; outch < outchannels; outch++) {
                            Quaternion sum = y[outch, ox, th];

                            for (int inch = 0; inch < inchannels; inch++) {
                                sum += x[inch, kx + ox * stride, th] * w[inch, outch, kx];
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
            int inchannels = 8, outchannels = 12, kwidth = 3, stride = 2, inwidth = 13, batch = 3;
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[batch * inwidth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            QuaternionMap1D x = new QuaternionMap1D(inchannels / 4, inwidth, batch, xcval);
            QuaternionFilter1D w = new QuaternionFilter1D(inchannels / 4, outchannels / 4, kwidth, wcval);

            QuaternionMap1D y = Reference(x, w, kwidth, stride);

            float[] y_expect = {
                -4.992000000e-03f,  3.696000000e-03f,  4.896000000e-03f,  4.116000000e-03f,  -3.744000000e-03f,  2.736000000e-03f,
                3.744000000e-03f,  3.060000000e-03f,  -2.496000000e-03f,  1.776000000e-03f,  2.592000000e-03f,  2.004000000e-03f,
                -1.305600000e-02f,  1.214400000e-02f,  1.353600000e-02f,  1.237200000e-02f,  -1.027200000e-02f,  9.648000000e-03f,
                1.084800000e-02f,  9.780000000e-03f,  -7.488000000e-03f,  7.152000000e-03f,  8.160000000e-03f,  7.188000000e-03f,
                -2.112000000e-02f,  2.059200000e-02f,  2.217600000e-02f,  2.062800000e-02f,  -1.680000000e-02f,  1.656000000e-02f,
                1.795200000e-02f,  1.650000000e-02f,  -1.248000000e-02f,  1.252800000e-02f,  1.372800000e-02f,  1.237200000e-02f,
                -2.918400000e-02f,  2.904000000e-02f,  3.081600000e-02f,  2.888400000e-02f,  -2.332800000e-02f,  2.347200000e-02f,
                2.505600000e-02f,  2.322000000e-02f,  -1.747200000e-02f,  1.790400000e-02f,  1.929600000e-02f,  1.755600000e-02f,
                -3.724800000e-02f,  3.748800000e-02f,  3.945600000e-02f,  3.714000000e-02f,  -2.985600000e-02f,  3.038400000e-02f,
                3.216000000e-02f,  2.994000000e-02f,  -2.246400000e-02f,  2.328000000e-02f,  2.486400000e-02f,  2.274000000e-02f,
                -4.531200000e-02f,  4.593600000e-02f,  4.809600000e-02f,  4.539600000e-02f,  -3.638400000e-02f,  3.729600000e-02f,
                3.926400000e-02f,  3.666000000e-02f,  -2.745600000e-02f,  2.865600000e-02f,  3.043200000e-02f,  2.792400000e-02f,
                -5.740800000e-02f,  5.860800000e-02f,  6.105600000e-02f,  5.778000000e-02f,  -4.617600000e-02f,  4.766400000e-02f,
                4.992000000e-02f,  4.674000000e-02f,  -3.494400000e-02f,  3.672000000e-02f,  3.878400000e-02f,  3.570000000e-02f,
                -6.547200000e-02f,  6.705600000e-02f,  6.969600000e-02f,  6.603600000e-02f,  -5.270400000e-02f,  5.457600000e-02f,
                5.702400000e-02f,  5.346000000e-02f,  -3.993600000e-02f,  4.209600000e-02f,  4.435200000e-02f,  4.088400000e-02f,
                -7.353600000e-02f,  7.550400000e-02f,  7.833600000e-02f,  7.429200000e-02f,  -5.923200000e-02f,  6.148800000e-02f,
                6.412800000e-02f,  6.018000000e-02f,  -4.492800000e-02f,  4.747200000e-02f,  4.992000000e-02f,  4.606800000e-02f,
                -8.160000000e-02f,  8.395200000e-02f,  8.697600000e-02f,  8.254800000e-02f,  -6.576000000e-02f,  6.840000000e-02f,
                7.123200000e-02f,  6.690000000e-02f,  -4.992000000e-02f,  5.284800000e-02f,  5.548800000e-02f,  5.125200000e-02f,
                -8.966400000e-02f,  9.240000000e-02f,  9.561600000e-02f,  9.080400000e-02f,  -7.228800000e-02f,  7.531200000e-02f,
                7.833600000e-02f,  7.362000000e-02f,  -5.491200000e-02f,  5.822400000e-02f,  6.105600000e-02f,  5.643600000e-02f,
                -9.772800000e-02f,  1.008480000e-01f,  1.042560000e-01f,  9.906000000e-02f,  -7.881600000e-02f,  8.222400000e-02f,
                8.544000000e-02f,  8.034000000e-02f,  -5.990400000e-02f,  6.360000000e-02f,  6.662400000e-02f,  6.162000000e-02f,
                -1.098240000e-01f,  1.135200000e-01f,  1.172160000e-01f,  1.114440000e-01f,  -8.860800000e-02f,  9.259200000e-02f,
                9.609600000e-02f,  9.042000000e-02f,  -6.739200000e-02f,  7.166400000e-02f,  7.497600000e-02f,  6.939600000e-02f,
                -1.178880000e-01f,  1.219680000e-01f,  1.258560000e-01f,  1.197000000e-01f,  -9.513600000e-02f,  9.950400000e-02f,
                1.032000000e-01f,  9.714000000e-02f,  -7.238400000e-02f,  7.704000000e-02f,  8.054400000e-02f,  7.458000000e-02f,
                -1.259520000e-01f,  1.304160000e-01f,  1.344960000e-01f,  1.279560000e-01f,  -1.016640000e-01f,  1.064160000e-01f,
                1.103040000e-01f,  1.038600000e-01f,  -7.737600000e-02f,  8.241600000e-02f,  8.611200000e-02f,  7.976400000e-02f,
                -1.340160000e-01f,  1.388640000e-01f,  1.431360000e-01f,  1.362120000e-01f,  -1.081920000e-01f,  1.133280000e-01f,
                1.174080000e-01f,  1.105800000e-01f,  -8.236800000e-02f,  8.779200000e-02f,  9.168000000e-02f,  8.494800000e-02f,
                -1.420800000e-01f,  1.473120000e-01f,  1.517760000e-01f,  1.444680000e-01f,  -1.147200000e-01f,  1.202400000e-01f,
                1.245120000e-01f,  1.173000000e-01f,  -8.736000000e-02f,  9.316800000e-02f,  9.724800000e-02f,  9.013200000e-02f,
                -1.501440000e-01f,  1.557600000e-01f,  1.604160000e-01f,  1.527240000e-01f,  -1.212480000e-01f,  1.271520000e-01f,
                1.316160000e-01f,  1.240200000e-01f,  -9.235200000e-02f,  9.854400000e-02f,  1.028160000e-01f,  9.531600000e-02f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");
        }
    }
}
