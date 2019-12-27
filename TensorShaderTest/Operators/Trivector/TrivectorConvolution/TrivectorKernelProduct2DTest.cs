using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.TrivectorConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorKernelProduct2DTest {
        [TestMethod]
        public void ExecuteTest() {
            Random random = new Random(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                    foreach (int outchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                        foreach (int kheight in new int[] { 1, 3, 5 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                    foreach (int inheight in new int[] { kheight, kheight * 2, 8, 9, 13, 17, 25 }) {
                                        int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                        float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                        float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                        float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                        Trivector[] xcval = (new Trivector[xval.Length / 3])
                                            .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

                                        Trivector[] ycval = (new Trivector[yval.Length / 3])
                                            .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

                                        Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                            .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                        TrivectorMap2D x = new TrivectorMap2D(inchannels / 3, inwidth, inheight, batch, xcval);
                                        TrivectorMap2D y = new TrivectorMap2D(outchannels / 3, outwidth, outheight, batch, ycval);
                                        Quaternion.QuaternionFilter2D w = new Quaternion.QuaternionFilter2D(inchannels / 3, outchannels / 3, kwidth, kheight, wcval);

                                        Quaternion.QuaternionFilter2D gw = Reference(x, y, w, kwidth, kheight);

                                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
                                        OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight), wval);

                                        OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight));

                                        TrivectorKernelProduct2D ope = new TrivectorKernelProduct2D(inwidth, inheight, inchannels, outchannels, kwidth, kheight, transpose: false, batch);

                                        ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

                                        float[] gw_expect = gw.ToArray();
                                        float[] gw_actual = gw_tensor.State;

                                        CollectionAssert.AreEqual(xval, x_tensor.State);
                                        CollectionAssert.AreEqual(yval, y_tensor.State);
                                        CollectionAssert.AreEqual(wval, w_tensor.State);

                                        AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

                                        Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

                                    }
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
            Random random = new Random(1234);

            float max_err = 0;

            int batch = 3;
            int inchannels = 147, outchannels = 150;
            int kwidth = 5, kheight = 3;
            int inwidth = 125, inheight = 196;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Trivector[] xcval = (new Trivector[xval.Length / 3])
                .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

            Trivector[] ycval = (new Trivector[yval.Length / 3])
                .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap2D x = new TrivectorMap2D(inchannels / 3, inwidth, inheight, batch, xcval);
            TrivectorMap2D y = new TrivectorMap2D(outchannels / 3, outwidth, outheight, batch, ycval);
            Quaternion.QuaternionFilter2D w = new Quaternion.QuaternionFilter2D(inchannels / 3, outchannels / 3, kwidth, kheight, wcval);

            Quaternion.QuaternionFilter2D gw = Reference(x, y, w, kwidth, kheight);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight), wval);

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight));

            TrivectorKernelProduct2D ope = new TrivectorKernelProduct2D(inwidth, inheight, inchannels, outchannels, kwidth, kheight, transpose: false, batch);

            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State;

            CollectionAssert.AreEqual(xval, x_tensor.State);
            CollectionAssert.AreEqual(yval, y_tensor.State);
            CollectionAssert.AreEqual(wval, w_tensor.State);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 32, inheight = 32, inchannels = 33, outchannels = 33, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, ksize, ksize));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, ksize, ksize));

            TrivectorKernelProduct2D ope = new TrivectorKernelProduct2D(inwidth, inheight, inchannels, outchannels, ksize, ksize);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/trivector_kernelproduct_2d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static Quaternion.QuaternionFilter2D Reference(TrivectorMap2D x, TrivectorMap2D gy, Quaternion.QuaternionFilter2D w, int kwidth, int kheight) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = gy.Width, outh = gy.Height;

            if (outw != inw - kwidth + 1 || outh != inh - kheight + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Quaternion.QuaternionFilter2D gw = new Quaternion.QuaternionFilter2D(inchannels, outchannels, kwidth, kheight);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int inch, outch = 0; outch < outchannels; outch++) {
                            for (inch = 0; inch < inchannels; inch++) {
                                Quaternion.Quaternion sum = 0;
                                Quaternion.Quaternion q = w[inch, outch, kx, ky];

                                for (int ix, iy = ky, ox, oy = 0; oy < outh; iy++, oy++) {
                                    for (ix = kx, ox = 0; ox < outw; ix++, ox++) {
                                        sum += Trivector.MulQGrad(x[inch, ix, iy, th], gy[outch, ox, oy, th], q);
                                    }
                                }

                                gw[inch, outch, kx, ky] += sum;
                            }
                        }
                    }

                }
            }

            return gw;
        }
    }
}
