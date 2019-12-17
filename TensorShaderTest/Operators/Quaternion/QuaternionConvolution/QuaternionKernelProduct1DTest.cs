using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.QuaternionConvolution;

namespace TensorShaderTest.Operators.Quaternion {
    [TestClass]
    public class QuaternionKernelProduct1DTest {
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
                                    float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                                        .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

                                    Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                                        .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

                                    QuaternionMap1D x = new QuaternionMap1D(inchannels / 4, inwidth, batch, xcval);
                                    QuaternionMap1D y = new QuaternionMap1D(outchannels / 4, outwidth, batch, ycval);

                                    QuaternionFilter1D gw = Reference(x, y, kwidth, stride);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);

                                    OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, kwidth));

                                    QuaternionKernelProduct1D ope = new QuaternionKernelProduct1D(inwidth, inchannels, outchannels, kwidth, stride, transpose: false, batch);

                                    ope.Execute(x_tensor, y_tensor, gw_tensor);

                                    float[] gw_expect = gw.ToArray();
                                    float[] gw_actual = gw_tensor.State;

                                    CollectionAssert.AreEqual(xval, x_tensor.State);
                                    CollectionAssert.AreEqual(yval, y_tensor.State);

                                    AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");

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
            foreach (bool transpose in new bool[] { false, true }) {
                foreach (int batch in new int[] { 1, 2, 3 }) {
                    foreach (int inchannels in new int[] { 4, 8, 12 }) {
                        foreach (int outchannels in new int[] { 4, 8, 12 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int stride in new int[] { 1, 2, 3 }) {
                                    foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                        int outwidth = inwidth - kwidth + 1;

                                        float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                        float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
                                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);

                                        OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, kwidth));

                                        QuaternionKernelProduct1D ope = new QuaternionKernelProduct1D(inwidth, inchannels, outchannels, kwidth, stride, transpose, batch);

                                        ope.Execute(x_tensor, y_tensor, gw_tensor);

                                        CollectionAssert.AreEqual(xval, x_tensor.State);
                                        CollectionAssert.AreEqual(yval, y_tensor.State);

                                        gw_tensor.CheckOverflow();

                                        Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch},{transpose}");

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
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, ksize));

            QuaternionKernelProduct1D ope = new QuaternionKernelProduct1D(inwidth, inchannels, outchannels, ksize, stride);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, y_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, gw_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static QuaternionFilter1D Reference(QuaternionMap1D x, QuaternionMap1D gy, int kwidth, int stride) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, outw = gy.Width;

            if (outw != inw - kwidth + 1) {
                throw new ArgumentException("mismatch shape");
            }

            QuaternionFilter1D w = new QuaternionFilter1D(inchannels, outchannels, kwidth);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int inch, outch = 0; outch < outchannels; outch++) {
                        for (inch = 0; inch < inchannels; inch++) {
                            Quaternion sum = 0;

                            for (int ix = kx, ox = 0; ox < outw; ix += stride, ox++) {
                                sum += Quaternion.MulGrad(gy[outch, ox, th], x[inch, ix, th]);
                            }

                            w[inch, outch, kx] += sum;
                        }
                    }

                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 8, outchannels = 12, kwidth = 3, stride = 2, inwidth = 13;
            int outwidth = inwidth - kwidth + 1, batch = 1;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            QuaternionMap1D x = new QuaternionMap1D(inchannels / 4, inwidth, batch, xcval);
            QuaternionMap1D y = new QuaternionMap1D(outchannels / 4, outwidth, batch, ycval);

            QuaternionFilter1D gw = Reference(x, y, kwidth, stride);

            float[] gw_expect = {
                2.587200000e-02f,  -0.000000000e+00f,  -1.944000000e-03f,  -9.720000000e-04f,  2.966400000e-02f,  -0.000000000e+00f,
                -2.040000000e-03f,  -1.020000000e-03f,  2.188800000e-02f,  -8.673617380e-19f,  -1.848000000e-03f,  -9.240000000e-04f,
                2.529600000e-02f,  8.673617380e-19f,  -1.944000000e-03f,  -9.720000000e-04f,  1.790400000e-02f,  -8.673617380e-19f,
                -1.752000000e-03f,  -8.760000000e-04f,  2.092800000e-02f,  -8.673617380e-19f,  -1.848000000e-03f,  -9.240000000e-04f,
                3.345600000e-02f,  -0.000000000e+00f,  -2.136000000e-03f,  -1.068000000e-03f,  3.724800000e-02f,  1.734723476e-18f,
                -2.232000000e-03f,  -1.116000000e-03f,  2.870400000e-02f,  -1.734723476e-18f,  -2.040000000e-03f,  -1.020000000e-03f,
                3.211200000e-02f,  1.734723476e-18f,  -2.136000000e-03f,  -1.068000000e-03f,  2.395200000e-02f,  -0.000000000e+00f,
                -1.944000000e-03f,  -9.720000000e-04f,  2.697600000e-02f,  -8.673617380e-19f,  -2.040000000e-03f,  -1.020000000e-03f,
                4.104000000e-02f,  -1.734723476e-18f,  -2.328000000e-03f,  -1.164000000e-03f,  4.483200000e-02f,  1.734723476e-18f,
                -2.424000000e-03f,  -1.212000000e-03f,  3.552000000e-02f,  -0.000000000e+00f,  -2.232000000e-03f,  -1.116000000e-03f,
                3.892800000e-02f,  -0.000000000e+00f,  -2.328000000e-03f,  -1.164000000e-03f,  3.000000000e-02f,  8.673617380e-19f,
                -2.136000000e-03f,  -1.068000000e-03f,  3.302400000e-02f,  3.469446952e-18f,  -2.232000000e-03f,  -1.116000000e-03f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");
        }
    }
}
