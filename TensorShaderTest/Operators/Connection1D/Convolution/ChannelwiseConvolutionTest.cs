using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class ChannelwiseConvolutionTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int kwidth in new int[] { 1, 3, 5 }) {
                        foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                            int outwidth = inwidth - kwidth + 1;

                            float[] xval = (new float[inwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[kwidth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map1D x = new Map1D(channels, inwidth, batch, xval);
                            Filter1D w = new Filter1D(channels, 1, kwidth, wval);

                            Map1D y = Reference(x, w, kwidth);

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth, batch), xval);
                            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(channels, 1, kwidth), wval);

                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth, batch));

                            ChannelwiseConvolution ope = new ChannelwiseConvolution(inwidth, channels, kwidth, batch);

                            ope.Execute(x_tensor, w_tensor, y_tensor);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_tensor.State;

                            CollectionAssert.AreEqual(xval, x_tensor.State);
                            CollectionAssert.AreEqual(wval, w_tensor.State);

                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{inwidth},{batch}");

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

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(channels, 1, ksize));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth));

            ChannelwiseConvolution ope = new ChannelwiseConvolution(inwidth, channels, ksize);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static Map1D Reference(Map1D x, Filter1D w, int kwidth) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, outw = inw - kwidth + 1;

            Map1D y = new Map1D(channels, outw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        for (int ch = 0; ch < channels; ch++) {
                            y[ch, ox, th] += x[ch, kx + ox, th] * w[ch, 0, kx];
                        }
                    }
                }
            }

            return y;
        }

        public static Map1D OptimizedReference(Map1D x, Filter1D w, int kwidth) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, outw = inw - kwidth + 1;

            Map1D y = new Map1D(channels, outw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                int inmap_offset = kx * channels;
                int kernel_offset = kx * channels;

                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        int inmap_idx = inmap_offset + ox * channels + th * inw * channels;
                        int outmap_idx = ox * channels + th * outw * channels;
                        int kernel_idx = kernel_offset;

                        for (int ch = 0; ch < channels; ch++) {
                            y[outmap_idx] += x[inmap_idx] * w[kernel_idx];

                            inmap_idx++;
                            outmap_idx++;
                            kernel_idx++;
                        }
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int channels = 7, kwidth = 3, inwidth = 13, batch = 2;
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[batch * inwidth * channels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map1D x = new Map1D(channels, inwidth, batch, xval);
            Filter1D w = new Filter1D(channels, 1, kwidth, wval);

            Map1D y = Reference(x, w, kwidth);

            float[] y_expect = {
                1.7500000e-04f,  1.9000000e-04f,  1.9900000e-04f,  2.0200000e-04f,  1.9900000e-04f,  1.9000000e-04f,  1.7500000e-04f,
                7.2100000e-04f,  6.9400000e-04f,  6.6100000e-04f,  6.2200000e-04f,  5.7700000e-04f,  5.2600000e-04f,  4.6900000e-04f,
                1.2670000e-03f,  1.1980000e-03f,  1.1230000e-03f,  1.0420000e-03f,  9.5500000e-04f,  8.6200000e-04f,  7.6300000e-04f,
                1.8130000e-03f,  1.7020000e-03f,  1.5850000e-03f,  1.4620000e-03f,  1.3330000e-03f,  1.1980000e-03f,  1.0570000e-03f,
                2.3590000e-03f,  2.2060000e-03f,  2.0470000e-03f,  1.8820000e-03f,  1.7110000e-03f,  1.5340000e-03f,  1.3510000e-03f,
                2.9050000e-03f,  2.7100000e-03f,  2.5090000e-03f,  2.3020000e-03f,  2.0890000e-03f,  1.8700000e-03f,  1.6450000e-03f,
                3.7240000e-03f,  3.4660000e-03f,  3.2020000e-03f,  2.9320000e-03f,  2.6560000e-03f,  2.3740000e-03f,  2.0860000e-03f,
                4.2700000e-03f,  3.9700000e-03f,  3.6640000e-03f,  3.3520000e-03f,  3.0340000e-03f,  2.7100000e-03f,  2.3800000e-03f,
                4.8160000e-03f,  4.4740000e-03f,  4.1260000e-03f,  3.7720000e-03f,  3.4120000e-03f,  3.0460000e-03f,  2.6740000e-03f,
                5.3620000e-03f,  4.9780000e-03f,  4.5880000e-03f,  4.1920000e-03f,  3.7900000e-03f,  3.3820000e-03f,  2.9680000e-03f,
                5.9080000e-03f,  5.4820000e-03f,  5.0500000e-03f,  4.6120000e-03f,  4.1680000e-03f,  3.7180000e-03f,  3.2620000e-03f,
                6.4540000e-03f,  5.9860000e-03f,  5.5120000e-03f,  5.0320000e-03f,  4.5460000e-03f,  4.0540000e-03f,  3.5560000e-03f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {channels},{kwidth},{inwidth},{batch}");
        }

        [TestMethod]
        public void OptimizeTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int kwidth in new int[] { 1, 3, 5 }) {
                        foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                            int outwidth = inwidth - kwidth + 1;

                            float[] xval = (new float[inwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[kwidth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map1D x = new Map1D(channels, inwidth, batch, xval);
                            Filter1D w = new Filter1D(channels, 1, kwidth, wval);

                            Map1D y = Reference(x, w, kwidth);
                            Map1D y_optimized = OptimizedReference(x, w, kwidth);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_optimized.ToArray();

                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{inwidth},{batch}");

                            Console.WriteLine($"pass: {channels},{kwidth},{inwidth},{batch}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }
    }
}
