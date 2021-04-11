using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class ChannelwiseKernelProductTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int kwidth in new int[] { 1, 3, 5 }) {
                        foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                            int outwidth = inwidth - kwidth + 1;

                            float[] xval = (new float[inwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] gyval = (new float[outwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map1D x = new(channels, inwidth, batch, xval);
                            Map1D gy = new(channels, outwidth, batch, gyval);

                            Filter1D gw = Reference(x, gy, kwidth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map1D(channels, inwidth, batch), xval);
                            OverflowCheckedTensor gy_tensor = new(Shape.Map1D(channels, outwidth, batch), gyval);

                            OverflowCheckedTensor gw_tensor = new(Shape.Kernel1D(channels, 1, kwidth));

                            ChannelwiseKernelProduct ope = new(inwidth, channels, kwidth, batch);

                            ope.Execute(x_tensor, gy_tensor, gw_tensor);

                            float[] gw_expect = gw.ToArray();
                            float[] gw_actual = gw_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                            CollectionAssert.AreEqual(gyval, gy_tensor.State.Value);

                            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{inwidth},{batch}");

                            Console.WriteLine($"pass: {channels},{kwidth},{inwidth},{batch}");

                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void LargeMapTest() {
            float max_err = 0;

            Random random = new(1234);

            int batch = 3;
            int channels = 49;
            int kwidth = 5;
            int inwidth = 125;
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[inwidth * channels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] gyval = (new float[outwidth * channels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map1D x = new(channels, inwidth, batch, xval);
            Map1D gy = new(channels, outwidth, batch, gyval);

            Filter1D gw = Reference(x, gy, kwidth);

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(channels, inwidth, batch), xval);
            OverflowCheckedTensor gy_tensor = new(Shape.Map1D(channels, outwidth, batch), gyval);

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel1D(channels, 1, kwidth));

            ChannelwiseKernelProduct ope = new(inwidth, channels, kwidth, batch);

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(gyval, gy_tensor.State.Value);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"pass: {channels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, channels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(channels, inwidth));
            OverflowCheckedTensor gy_tensor = new(Shape.Map1D(channels, outwidth));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel1D(channels, 1, ksize));

            ChannelwiseKernelProduct ope = new(inwidth, channels, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/chwise_kernelproduct_1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static Filter1D Reference(Map1D x, Map1D gy, int kwidth) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, outw = gy.Width;

            if (outw != inw - kwidth + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter1D w = new(channels, 1, kwidth);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ix = kx, ox = 0; ox < outw; ix++, ox++) {
                        for (int ch = 0; ch < channels; ch++) {
                            w[ch, 0, kx] += x[ch, ix, th] * gy[ch, ox, th];
                        }
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int channels = 7, kwidth = 3, inwidth = 13;
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[inwidth * channels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] gyval = (new float[outwidth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map1D x = new(channels, inwidth, 1, xval);
            Map1D gy = new(channels, outwidth, 1, gyval);

            Filter1D gw = Reference(x, gy, kwidth);

            float[] gw_expect = {
                1.03949998e-02f,  1.04499999e-02f,  1.04830004e-02f,  1.04940003e-02f,  1.04830004e-02f,  1.04499999e-02f,  1.03949998e-02f,
                1.35519998e-02f,  1.35300001e-02f,  1.34859998e-02f,  1.34199997e-02f,  1.33320000e-02f,  1.32219996e-02f,  1.30899996e-02f,
                1.67089999e-02f,  1.66100003e-02f,  1.64889991e-02f,  1.63460001e-02f,  1.61809996e-02f,  1.59939993e-02f,  1.57849994e-02f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {channels},{kwidth},{inwidth}");
        }
    }
}
