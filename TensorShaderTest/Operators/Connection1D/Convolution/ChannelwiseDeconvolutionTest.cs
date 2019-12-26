using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class ChannelwiseDeconvolutionTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int kwidth in new int[] { 1, 3, 5 }) {
                        foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                            int outwidth = inwidth - kwidth + 1;

                            float[] yval = (new float[outwidth * channels * batch]).Select((_, idx) => idx * 1e-4f).ToArray();
                            float[] wval = (new float[kwidth * channels]).Select((_, idx) => idx * 1e-4f).Reverse().ToArray();

                            Map1D y = new Map1D(channels, outwidth, batch, yval);
                            Filter1D w = new Filter1D(channels, 1, kwidth, wval);

                            Map1D x = Reference(y, w, inwidth, kwidth);

                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth, batch), yval);
                            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(channels, 1, kwidth), wval);

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth, batch));

                            ChannelwiseDeconvolution ope = new ChannelwiseDeconvolution(outwidth, channels, kwidth, batch);

                            ope.Execute(y_tensor, w_tensor, x_tensor);

                            float[] x_expect = x.ToArray();
                            float[] x_actual = x_tensor.State;

                            CollectionAssert.AreEqual(yval, y_tensor.State);
                            CollectionAssert.AreEqual(wval, w_tensor.State);

                            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{inwidth},{batch}");

                            Console.WriteLine($"pass: {channels},{kwidth},{inwidth},{batch}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, channels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(channels, 1, ksize));

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth));

            ChannelwiseDeconvolution ope = new ChannelwiseDeconvolution(outwidth, channels, ksize);

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

        public static Map1D Reference(Map1D y, Filter1D w, int inw, int kwidth) {
            int channels = w.InChannels, batch = y.Batch;
            int outw = inw - kwidth + 1;

            if (y.Width != outw) {
                throw new ArgumentException("mismatch shape");
            }

            Map1D x = new Map1D(channels, inw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        for (int ch = 0; ch < channels; ch++) {
                            x[ch, kx + ox, th] += y[ch, ox, th] * w[ch, 0, kx];
                        }
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int channels = 7, kwidth = 3, inwidth = 13, batch = 2;
            int outwidth = inwidth - kwidth + 1;

            float[] yval = (new float[outwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map1D y = new Map1D(channels, outwidth, batch, yval);
            Filter1D w = new Filter1D(channels, 1, kwidth, wval);

            Map1D x = Reference(y, w, inwidth, kwidth);

            float[] x_expect = {
                0.0000e+00f,  1.9000e-05f,  3.6000e-05f,  5.1000e-05f,  6.4000e-05f,  7.5000e-05f,  8.4000e-05f,
                0.0000e+00f,  1.2000e-05f,  2.2000e-05f,  3.0000e-05f,  3.6000e-05f,  4.0000e-05f,  4.2000e-05f,
                1.4000e-04f,  1.5700e-04f,  1.7000e-04f,  1.7900e-04f,  1.8400e-04f,  1.8500e-04f,  1.8200e-04f,
                9.1000e-05f,  9.6000e-05f,  9.9000e-05f,  1.0000e-04f,  9.9000e-05f,  9.6000e-05f,  9.1000e-05f,
                3.2200e-04f,  3.2500e-04f,  3.2400e-04f,  3.1900e-04f,  3.1000e-04f,  2.9700e-04f,  2.8000e-04f,
                1.8200e-04f,  1.8000e-04f,  1.7600e-04f,  1.7000e-04f,  1.6200e-04f,  1.5200e-04f,  1.4000e-04f,
                5.0400e-04f,  4.9300e-04f,  4.7800e-04f,  4.5900e-04f,  4.3600e-04f,  4.0900e-04f,  3.7800e-04f,
                2.7300e-04f,  2.6400e-04f,  2.5300e-04f,  2.4000e-04f,  2.2500e-04f,  2.0800e-04f,  1.8900e-04f,
                6.8600e-04f,  6.6100e-04f,  6.3200e-04f,  5.9900e-04f,  5.6200e-04f,  5.2100e-04f,  4.7600e-04f,
                3.6400e-04f,  3.4800e-04f,  3.3000e-04f,  3.1000e-04f,  2.8800e-04f,  2.6400e-04f,  2.3800e-04f,
                8.6800e-04f,  8.2900e-04f,  7.8600e-04f,  7.3900e-04f,  6.8800e-04f,  6.3300e-04f,  5.7400e-04f,
                4.5500e-04f,  4.3200e-04f,  4.0700e-04f,  3.8000e-04f,  3.5100e-04f,  3.2000e-04f,  2.8700e-04f,
                2.1000e-04f,  1.8000e-04f,  1.4800e-04f,  1.1400e-04f,  7.8000e-05f,  4.0000e-05f,  0.0000e+00f,
                8.4000e-04f,  8.1700e-04f,  7.9200e-04f,  7.6500e-04f,  7.3600e-04f,  7.0500e-04f,  6.7200e-04f,
                5.4600e-04f,  5.1600e-04f,  4.8400e-04f,  4.5000e-04f,  4.1400e-04f,  3.7600e-04f,  3.3600e-04f,
                1.2320e-03f,  1.1650e-03f,  1.0940e-03f,  1.0190e-03f,  9.4000e-04f,  8.5700e-04f,  7.7000e-04f,
                6.3700e-04f,  6.0000e-04f,  5.6100e-04f,  5.2000e-04f,  4.7700e-04f,  4.3200e-04f,  3.8500e-04f,
                1.4140e-03f,  1.3330e-03f,  1.2480e-03f,  1.1590e-03f,  1.0660e-03f,  9.6900e-04f,  8.6800e-04f,
                7.2800e-04f,  6.8400e-04f,  6.3800e-04f,  5.9000e-04f,  5.4000e-04f,  4.8800e-04f,  4.3400e-04f,
                1.5960e-03f,  1.5010e-03f,  1.4020e-03f,  1.2990e-03f,  1.1920e-03f,  1.0810e-03f,  9.6600e-04f,
                8.1900e-04f,  7.6800e-04f,  7.1500e-04f,  6.6000e-04f,  6.0300e-04f,  5.4400e-04f,  4.8300e-04f,
                1.7780e-03f,  1.6690e-03f,  1.5560e-03f,  1.4390e-03f,  1.3180e-03f,  1.1930e-03f,  1.0640e-03f,
                9.1000e-04f,  8.5200e-04f,  7.9200e-04f,  7.3000e-04f,  6.6600e-04f,  6.0000e-04f,  5.3200e-04f,
                1.9600e-03f,  1.8370e-03f,  1.7100e-03f,  1.5790e-03f,  1.4440e-03f,  1.3050e-03f,  1.1620e-03f,
                1.0010e-03f,  9.3600e-04f,  8.6900e-04f,  8.0000e-04f,  7.2900e-04f,  6.5600e-04f,  5.8100e-04f,
                4.6200e-04f,  3.9000e-04f,  3.1600e-04f,  2.4000e-04f,  1.6200e-04f,  8.2000e-05f,  0.0000e+00f,
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {channels},{kwidth},{inwidth},{batch}");
        }
    }
}
