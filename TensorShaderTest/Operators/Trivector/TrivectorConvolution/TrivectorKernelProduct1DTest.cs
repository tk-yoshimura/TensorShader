using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.TrivectorConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorKernelProduct1DTest {
        [TestMethod]
        public void ExecuteTest() {
            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                    foreach (int outchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                int outwidth = inwidth - kwidth + 1;

                                float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                float[] wval = (new float[kwidth * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                Trivector[] xcval = (new Trivector[xval.Length / 3])
                                    .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

                                Trivector[] ycval = (new Trivector[yval.Length / 3])
                                    .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

                                Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                    .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                TrivectorMap1D x = new(inchannels / 3, inwidth, batch, xcval);
                                TrivectorMap1D y = new(outchannels / 3, outwidth, batch, ycval);
                                Quaternion.QuaternionFilter1D w = new(inchannels / 3, outchannels / 3, kwidth, wcval);

                                Quaternion.QuaternionFilter1D gw = Reference(x, y, w, kwidth);

                                OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch), xval);
                                OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth, batch), yval);
                                OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth), wval);

                                OverflowCheckedTensor gw_tensor = new(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth));

                                TrivectorKernelProduct1D ope = new(inwidth, inchannels, outchannels, kwidth, transpose: false, batch);

                                ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

                                float[] gw_expect = gw.ToArray();
                                float[] gw_actual = gw_tensor.State.Value;

                                CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                                CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void LargeMapTest() {
            Random random = new(1234);

            float max_err = 0;

            int batch = 3;
            int inchannels = 147, outchannels = 150;
            int kwidth = 5;
            int inwidth = 125;
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Trivector[] xcval = (new Trivector[xval.Length / 3])
                .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

            Trivector[] ycval = (new Trivector[yval.Length / 3])
                .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap1D x = new(inchannels / 3, inwidth, batch, xcval);
            TrivectorMap1D y = new(outchannels / 3, outwidth, batch, ycval);
            Quaternion.QuaternionFilter1D w = new(inchannels / 3, outchannels / 3, kwidth, wcval);

            Quaternion.QuaternionFilter1D gw = Reference(x, y, w, kwidth);

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch), xval);
            OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth, batch), yval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth), wval);

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth));

            TrivectorKernelProduct1D ope = new(inwidth, inchannels, outchannels, kwidth, transpose: false, batch);

            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 32, inchannels = 33, outchannels = 33, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, ksize));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, ksize));

            TrivectorKernelProduct1D ope = new(inwidth, inchannels, outchannels, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/trivector_kernelproduct_1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static Quaternion.QuaternionFilter1D Reference(TrivectorMap1D x, TrivectorMap1D gy, Quaternion.QuaternionFilter1D w, int kwidth) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, outw = gy.Width;

            if (outw != inw - kwidth + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Quaternion.QuaternionFilter1D gw = new(inchannels, outchannels, kwidth);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ix = kx, ox = 0; ox < outw; ix++, ox++) {
                        for (int inch, outch = 0; outch < outchannels; outch++) {
                            for (inch = 0; inch < inchannels; inch++) {
                                gw[inch, outch, kx] += Trivector.MulQGrad(x[inch, ix, th], gy[outch, ox, th], w[inch, outch, kx]);
                            }
                        }
                    }
                }
            }

            return gw;
        }
    }
}
