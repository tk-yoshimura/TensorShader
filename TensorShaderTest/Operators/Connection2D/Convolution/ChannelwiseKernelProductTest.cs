using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class ChannelwiseKernelProductTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int kheight in new int[] { 1, 3, 5 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                    int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                    float[] xval = (new float[inwidth * inheight * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] gyval = (new float[outwidth * outheight * channels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    Map2D x = new Map2D(channels, inwidth, inheight, batch, xval);
                                    Map2D gy = new Map2D(channels, outwidth, outheight, batch, gyval);

                                    Filter2D gw = Reference(x, gy, kwidth, kheight);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight, batch), xval);
                                    OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight, batch), gyval);

                                    OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel2D(channels, 1, kwidth, kheight));

                                    ChannelwiseKernelProduct ope = new ChannelwiseKernelProduct(inwidth, inheight, channels, kwidth, kheight, batch);

                                    ope.Execute(x_tensor, gy_tensor, gw_tensor);

                                    float[] gw_expect = gw.ToArray();
                                    float[] gw_actual = gw_tensor.State;

                                    CollectionAssert.AreEqual(xval, x_tensor.State);
                                    CollectionAssert.AreEqual(gyval, gy_tensor.State);

                                    AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

                                    Console.WriteLine($"pass: {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void LargeMapTest() {
            float max_err = 0;

            Random random = new Random(1234);

            int batch = 3;
            int channels = 49;
            int kwidth = 5, kheight = 3;
            int inwidth = 250, inheight = 196;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[inwidth * inheight * channels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] gyval = (new float[outwidth * outheight * channels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map2D x = new Map2D(channels, inwidth, inheight, batch, xval);
            Map2D gy = new Map2D(channels, outwidth, outheight, batch, gyval);

            Filter2D gw = Reference(x, gy, kwidth, kheight);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight, batch), xval);
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight, batch), gyval);

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel2D(channels, 1, kwidth, kheight));

            ChannelwiseKernelProduct ope = new ChannelwiseKernelProduct(inwidth, inheight, channels, kwidth, kheight, batch);

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State;

            CollectionAssert.AreEqual(xval, x_tensor.State);
            CollectionAssert.AreEqual(gyval, gy_tensor.State);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"pass: {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inheight = 512, channels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight));
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel2D(channels, 1, ksize, ksize));

            ChannelwiseKernelProduct ope = new ChannelwiseKernelProduct(inwidth, inheight, channels, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/chwise_kernelproduct_2d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static Filter2D Reference(Map2D x, Map2D gy, int kwidth, int kheight) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = gy.Width, outh = gy.Height;

            if (outw != inw - kwidth + 1 || outh != inh - kheight + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter2D w = new Filter2D(channels, 1, kwidth, kheight);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ix, iy = ky, ox, oy = 0; oy < outh; iy++, oy++) {
                            for (ix = kx, ox = 0; ox < outw; ix++, ox++) {
                                for (int ch = 0; ch < channels; ch++) {
                                    w[ch, 0, kx, ky] += x[ch, ix, iy, th] * gy[ch, ox, oy, th];
                                }
                            }
                        }
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int channels = 7, kwidth = 3, kheight = 5, inwidth = 13, inheight = 17;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[inwidth * inheight * channels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] gyval = (new float[outwidth * outheight * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map2D x = new Map2D(channels, inwidth, inheight, 1, xval);
            Map2D gy = new Map2D(channels, outwidth, outheight, 1, gyval);

            Filter2D gw = Reference(x, gy, kwidth, kheight);

            float[] gw_expect = {
                2.76926650e+01f,  2.76813680e+01f,  2.76697850e+01f,  2.76579160e+01f,  2.76457610e+01f,  2.76333200e+01f,  2.76205930e+01f,
                2.81961680e+01f,  2.81838700e+01f,  2.81712860e+01f,  2.81584160e+01f,  2.81452600e+01f,  2.81318180e+01f,  2.81180900e+01f,
                2.86996710e+01f,  2.86863720e+01f,  2.86727870e+01f,  2.86589160e+01f,  2.86447590e+01f,  2.86303160e+01f,  2.86155870e+01f,
                3.42382040e+01f,  3.42138940e+01f,  3.41892980e+01f,  3.41644160e+01f,  3.41392480e+01f,  3.41137940e+01f,  3.40880540e+01f,
                3.47417070e+01f,  3.47163960e+01f,  3.46907990e+01f,  3.46649160e+01f,  3.46387470e+01f,  3.46122920e+01f,  3.45855510e+01f,
                3.52452100e+01f,  3.52188980e+01f,  3.51923000e+01f,  3.51654160e+01f,  3.51382460e+01f,  3.51107900e+01f,  3.50830480e+01f,
                4.07837430e+01f,  4.07464200e+01f,  4.07088110e+01f,  4.06709160e+01f,  4.06327350e+01f,  4.05942680e+01f,  4.05555150e+01f,
                4.12872460e+01f,  4.12489220e+01f,  4.12103120e+01f,  4.11714160e+01f,  4.11322340e+01f,  4.10927660e+01f,  4.10530120e+01f,
                4.17907490e+01f,  4.17514240e+01f,  4.17118130e+01f,  4.16719160e+01f,  4.16317330e+01f,  4.15912640e+01f,  4.15505090e+01f,
                4.73292820e+01f,  4.72789460e+01f,  4.72283240e+01f,  4.71774160e+01f,  4.71262220e+01f,  4.70747420e+01f,  4.70229760e+01f,
                4.78327850e+01f,  4.77814480e+01f,  4.77298250e+01f,  4.76779160e+01f,  4.76257210e+01f,  4.75732400e+01f,  4.75204730e+01f,
                4.83362880e+01f,  4.82839500e+01f,  4.82313260e+01f,  4.81784160e+01f,  4.81252200e+01f,  4.80717380e+01f,  4.80179700e+01f,
                5.38748210e+01f,  5.38114720e+01f,  5.37478370e+01f,  5.36839160e+01f,  5.36197090e+01f,  5.35552160e+01f,  5.34904370e+01f,
                5.43783240e+01f,  5.43139740e+01f,  5.42493380e+01f,  5.41844160e+01f,  5.41192080e+01f,  5.40537140e+01f,  5.39879340e+01f,
                5.48818270e+01f,  5.48164760e+01f,  5.47508390e+01f,  5.46849160e+01f,  5.46187070e+01f,  5.45522120e+01f,  5.44854310e+01f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {channels},{kwidth},{kheight},{inwidth},{inheight}");
        }
    }
}
