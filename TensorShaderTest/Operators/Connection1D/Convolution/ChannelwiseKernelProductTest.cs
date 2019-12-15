using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class ChannelwiseKernelProductTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int kwidth in new int[] { 1, 3, 5 }) {
                        foreach (int stride in new int[] { 1, 2, 3 }) {
                            foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                int outwidth = (inwidth - kwidth) / stride + 1;

                                float[] xval = (new float[inwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] gyval = (new float[outwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                Map1D x = new Map1D(channels, inwidth, batch, xval);
                                Map1D gy = new Map1D(channels, outwidth, batch, gyval);

                                Filter1D gw = Reference(x, gy, kwidth, stride);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth, batch), xval);
                                OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth, batch), gyval);

                                OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel1D(channels, 1, kwidth));

                                ChannelwiseKernelProduct ope = new ChannelwiseKernelProduct(inwidth, channels, kwidth, stride, batch);

                                ope.Execute(x_tensor, gy_tensor, gw_tensor);

                                float[] gw_expect = gw.ToArray();
                                float[] gw_actual = gw_tensor.State;

                                CollectionAssert.AreEqual(xval, x_tensor.State);
                                CollectionAssert.AreEqual(gyval, gy_tensor.State);

                                AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{stride},{inwidth},{batch}");

                                Console.WriteLine($"pass: {channels},{kwidth},{stride},{inwidth},{batch}");

                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, channels = 32, ksize = 3, stride = 2;
            int outwidth = (inwidth - ksize) / stride + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth));
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel1D(channels, 1, ksize));

            ChannelwiseKernelProduct ope = new ChannelwiseKernelProduct(inwidth, channels, ksize, stride);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);
            ope.Execute(x_tensor, gy_tensor, gw_tensor);
            ope.Execute(x_tensor, gy_tensor, gw_tensor);
            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static Filter1D Reference(Map1D x, Map1D gy, int kwidth, int stride) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, outw = gy.Width;

            if (outw != (inw - kwidth) / stride + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter1D w = new Filter1D(channels, 1, kwidth);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ch = 0; ch < channels; ch++) {
                        double sum = 0;

                        for (int ix = kx, ox = 0; ox < outw; ix += stride, ox++) {
                            sum += x[ch, ix, th] * gy[ch, ox, th];
                        }

                        w[ch, 0, kx] += sum;
                    }

                }
            }

            return w;
        }

        public static Filter1D OptimizedReference(Map1D x, Map1D gy, int kwidth, int stride) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, outw = gy.Width;

            if (outw < (inw - kwidth) / stride + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter1D w = new Filter1D(channels, 1, kwidth);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ch = 0; ch < channels; ch++) {
                        int filter_idx = ch + kx * channels;
                        int inmap_idx = ch + kx * channels + th * inw * channels;
                        int outmap_idx = ch + th * outw * channels;

                        double sum = 0;

                        for (int ox = 0; ox < outw; ox++) {
                            sum += x[inmap_idx] * gy[outmap_idx];

                            inmap_idx += channels * stride;
                            outmap_idx += channels;
                        }

                        w[filter_idx] += sum;
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int channels = 7, kwidth = 3, stride = 2, inwidth = 13;
            int outwidth = (inwidth - kwidth) / stride + 1;

            float[] xval = (new float[inwidth * channels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] gyval = (new float[outwidth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map1D x = new Map1D(channels, inwidth, 1, xval);
            Map1D gy = new Map1D(channels, outwidth, 1, gyval);

            Filter1D gw = Reference(x, gy, kwidth, stride);

            float[] gw_expect = {
                3.22000007e-03f,  3.14499997e-03f,  3.05800000e-03f,  2.95899995e-03f,  2.84800003e-03f,  2.72500003e-03f,  2.58999993e-03f,
                4.20700014e-03f,  4.08999994e-03f,  3.96100013e-03f,  3.81999998e-03f,  3.66699998e-03f,  3.50199989e-03f,  3.32499994e-03f,
                5.19399997e-03f,  5.03499992e-03f,  4.86399978e-03f,  4.68100002e-03f,  4.48600017e-03f,  4.27900022e-03f,  4.06000018e-03f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {channels},{kwidth},{stride},{inwidth}");
        }

        [TestMethod]
        public void OptimizeTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int kwidth in new int[] { 1, 3, 5 }) {
                        foreach (int stride in new int[] { 1, 2, 3 }) {
                            foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                int outwidth = (inwidth - kwidth) / stride + 1;

                                float[] xval = (new float[inwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] gyval = (new float[outwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                Map1D x = new Map1D(channels, inwidth, batch, xval);
                                Map1D gy = new Map1D(channels, outwidth, batch, gyval);

                                Filter1D gw = Reference(x, gy, kwidth, stride);
                                Filter1D gw_optimized = OptimizedReference(x, gy, kwidth, stride);

                                float[] gw_expect = gw.ToArray();
                                float[] gw_actual = gw_optimized.ToArray();

                                AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{stride},{inwidth},{batch}");

                                Console.WriteLine($"pass: {channels},{kwidth},{stride},{inwidth},{batch}");
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }
    }
}
