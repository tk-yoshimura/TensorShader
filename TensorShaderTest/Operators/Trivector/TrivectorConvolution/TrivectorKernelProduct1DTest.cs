using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.TrivectorConvolution;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorKernelProduct1DTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int inchannels in new int[] { 3, 6, 9, 12 }) {
                    foreach (int outchannels in new int[] { 3, 6, 9, 12 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int stride in new int[] { 1, 2, 3 }) {
                                foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                    int outwidth = (inwidth - kwidth) / stride + 1;

                                    float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();
                                    float[] wval = (new float[kwidth * inchannels * outchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    Trivector[] xcval = (new Trivector[xval.Length / 3])
                                        .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

                                    Trivector[] ycval = (new Trivector[yval.Length / 3])
                                        .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

                                    Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                        .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                    TrivectorMap1D x = new TrivectorMap1D(inchannels / 3, inwidth, batch, xcval);
                                    TrivectorMap1D y = new TrivectorMap1D(outchannels / 3, outwidth, batch, ycval);
                                    Quaternion.QuaternionFilter1D w = new Quaternion.QuaternionFilter1D(inchannels / 3, outchannels / 3, kwidth, wcval);

                                    Quaternion.QuaternionFilter1D gw = Reference(x, y, w, kwidth, stride);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);
                                    OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth), wval);

                                    OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth));

                                    TrivectorKernelProduct1D ope = new TrivectorKernelProduct1D(inwidth, inchannels, outchannels, kwidth, stride, transpose: false, batch);

                                    ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

                                    float[] gw_expect = gw.ToArray();
                                    float[] gw_actual = gw_tensor.State;

                                    CollectionAssert.AreEqual(xval, x_tensor.State);
                                    CollectionAssert.AreEqual(yval, y_tensor.State);
                                    CollectionAssert.AreEqual(wval, w_tensor.State);

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
                    foreach (int inchannels in new int[] { 3, 6, 9, 12 }) {
                        foreach (int outchannels in new int[] { 3, 6, 9, 12 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int stride in new int[] { 1, 2, 3 }) {
                                    foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                        int outwidth = (inwidth - kwidth) / stride + 1;

                                        float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                        float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();
                                        float[] wval = (new float[kwidth * inchannels * outchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
                                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);
                                        OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth), wval);

                                        OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth));

                                        TrivectorKernelProduct1D ope = new TrivectorKernelProduct1D(inwidth, inchannels, outchannels, kwidth, stride, transpose, batch);

                                        ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

                                        CollectionAssert.AreEqual(xval, x_tensor.State);
                                        CollectionAssert.AreEqual(yval, y_tensor.State);
                                        CollectionAssert.AreEqual(wval, w_tensor.State);

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
            int inwidth = 32, inchannels = 33, outchannels = 33, ksize = 3, stride = 2;
            int outwidth = (inwidth - ksize) / stride + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, ksize));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, ksize));

            TrivectorKernelProduct1D ope = new TrivectorKernelProduct1D(inwidth, inchannels, outchannels, ksize, stride);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static Quaternion.QuaternionFilter1D Reference(TrivectorMap1D x, TrivectorMap1D gy, Quaternion.QuaternionFilter1D w, int kwidth, int stride) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, outw = gy.Width;

            if (outw != (inw - kwidth) / stride + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Quaternion.QuaternionFilter1D gw = new Quaternion.QuaternionFilter1D(inchannels, outchannels, kwidth);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int inch, outch = 0; outch < outchannels; outch++) {
                        for (inch = 0; inch < inchannels; inch++) {
                            Quaternion.Quaternion sum = 0;
                            Quaternion.Quaternion q = w[inch, outch, kx];

                            for (int ix = kx, ox = 0; ox < outw; ix += stride, ox++) {
                                sum += Trivector.MulQGrad(x[inch, ix, th], gy[outch, ox, th], q);
                            }

                            gw[inch, outch, kx] += sum;
                        }
                    }
                }

            }

            return gw;
        }
    }
}
