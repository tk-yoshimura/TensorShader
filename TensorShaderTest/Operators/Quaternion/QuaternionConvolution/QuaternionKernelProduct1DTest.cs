using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.QuaternionConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Quaternion {
    [TestClass]
    public class QuaternionKernelProduct1DTest {
        [TestMethod]
        public void ExecuteTest() {
            Random random = new Random(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 4, 8, 20, 32, 36 }) {
                    foreach (int outchannels in new int[] { 4, 8, 20, 32, 36 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {

                                int outwidth = inwidth - kwidth + 1;

                                float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                                    .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

                                Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                                    .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

                                QuaternionMap1D x = new QuaternionMap1D(inchannels / 4, inwidth, batch, xcval);
                                QuaternionMap1D y = new QuaternionMap1D(outchannels / 4, outwidth, batch, ycval);

                                QuaternionFilter1D gw = Reference(x, y, kwidth);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);

                                OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, kwidth));

                                QuaternionKernelProduct1D ope = new QuaternionKernelProduct1D(inwidth, inchannels, outchannels, kwidth, transpose: false, batch);

                                ope.Execute(x_tensor, y_tensor, gw_tensor);

                                float[] gw_expect = gw.ToArray();
                                float[] gw_actual = gw_tensor.State.Value;

                                CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                CollectionAssert.AreEqual(yval, y_tensor.State.Value);

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
            Random random = new Random(1234);

            float max_err = 0;

            int batch = 3;
            int inchannels = 196, outchannels = 200;
            int kwidth = 5;
            int inwidth = 125;
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            QuaternionMap1D x = new QuaternionMap1D(inchannels / 4, inwidth, batch, xcval);
            QuaternionMap1D y = new QuaternionMap1D(outchannels / 4, outwidth, batch, ycval);

            QuaternionFilter1D gw = Reference(x, y, kwidth);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, kwidth));

            QuaternionKernelProduct1D ope = new QuaternionKernelProduct1D(inwidth, inchannels, outchannels, kwidth, transpose: false, batch);

            ope.Execute(x_tensor, y_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(yval, y_tensor.State.Value);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, ksize));

            QuaternionKernelProduct1D ope = new QuaternionKernelProduct1D(inwidth, inchannels, outchannels, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/quaternion_kernelproduct_1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static QuaternionFilter1D Reference(QuaternionMap1D x, QuaternionMap1D gy, int kwidth) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, outw = gy.Width;

            if (outw != inw - kwidth + 1) {
                throw new ArgumentException("mismatch shape");
            }

            QuaternionFilter1D w = new QuaternionFilter1D(inchannels, outchannels, kwidth);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ix = kx, ox = 0; ox < outw; ix++, ox++) {
                        for (int inch, outch = 0; outch < outchannels; outch++) {
                            for (inch = 0; inch < inchannels; inch++) {
                                w[inch, outch, kx] += Quaternion.MulGrad(gy[outch, ox, th], x[inch, ix, th]);
                            }
                        }
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 8, outchannels = 12, kwidth = 3, inwidth = 13;
            int outwidth = inwidth - kwidth + 1, batch = 1;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            QuaternionMap1D x = new QuaternionMap1D(inchannels / 4, inwidth, batch, xcval);
            QuaternionMap1D y = new QuaternionMap1D(outchannels / 4, outwidth, batch, ycval);

            QuaternionFilter1D gw = Reference(x, y, kwidth);

            float[] gw_expect = {
                8.461200000e-02f,  -0.000000000e+00f,  -4.884000000e-03f,  -2.442000000e-03f,  9.684400000e-02f,  3.469446952e-18f,
                -5.060000000e-03f,  -2.530000000e-03f,  7.730800000e-02f,  3.469446952e-18f,  -4.708000000e-03f,  -2.354000000e-03f,
                8.883600000e-02f,  6.938893904e-18f,  -4.884000000e-03f,  -2.442000000e-03f,  7.000400000e-02f,  3.469446952e-18f,
                -4.532000000e-03f,  -2.266000000e-03f,  8.082800000e-02f,  3.469446952e-18f,  -4.708000000e-03f,  -2.354000000e-03f,
                1.090760000e-01f,  -0.000000000e+00f,  -5.236000000e-03f,  -2.618000000e-03f,  1.213080000e-01f,  3.469446952e-18f,
                -5.412000000e-03f,  -2.706000000e-03f,  1.003640000e-01f,  -0.000000000e+00f,  -5.060000000e-03f,  -2.530000000e-03f,
                1.118920000e-01f,  6.938893904e-18f,  -5.236000000e-03f,  -2.618000000e-03f,  9.165200000e-02f,  -0.000000000e+00f,
                -4.884000000e-03f,  -2.442000000e-03f,  1.024760000e-01f,  -0.000000000e+00f,  -5.060000000e-03f,  -2.530000000e-03f,
                1.335400000e-01f,  -0.000000000e+00f,  -5.588000000e-03f,  -2.794000000e-03f,  1.457720000e-01f,  -6.938893904e-18f,
                -5.764000000e-03f,  -2.882000000e-03f,  1.234200000e-01f,  -6.938893904e-18f,  -5.412000000e-03f,  -2.706000000e-03f,
                1.349480000e-01f,  6.938893904e-18f,  -5.588000000e-03f,  -2.794000000e-03f,  1.133000000e-01f,  3.469446952e-18f,
                -5.236000000e-03f,  -2.618000000e-03f,  1.241240000e-01f,  -6.938893904e-18f,  -5.412000000e-03f,  -2.706000000e-03f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");
        }
    }
}
