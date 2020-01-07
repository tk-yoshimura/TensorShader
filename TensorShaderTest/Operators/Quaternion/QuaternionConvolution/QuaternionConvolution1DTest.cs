using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.QuaternionConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Quaternion {
    [TestClass]
    public class QuaternionConvolution1DTest {
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
                                float[] wval = (new float[kwidth * inchannels * outchannels / 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                                    .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

                                Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                                    .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                QuaternionMap1D x = new QuaternionMap1D(inchannels / 4, inwidth, batch, xcval);
                                QuaternionFilter1D w = new QuaternionFilter1D(inchannels / 4, outchannels / 4, kwidth, wcval);

                                QuaternionMap1D y = Reference(x, w, kwidth);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
                                OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, kwidth), wval);

                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch));

                                QuaternionConvolution1D ope = new QuaternionConvolution1D(inwidth, inchannels, outchannels, kwidth, gradmode: false, batch);

                                ope.Execute(x_tensor, w_tensor, y_tensor);

                                float[] y_expect = y.ToArray();
                                float[] y_actual = y_tensor.State;

                                CollectionAssert.AreEqual(xval, x_tensor.State);
                                CollectionAssert.AreEqual(wval, w_tensor.State);

                                AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

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
            float[] wval = (new float[kwidth * inchannels * outchannels / 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            QuaternionMap1D x = new QuaternionMap1D(inchannels / 4, inwidth, batch, xcval);
            QuaternionFilter1D w = new QuaternionFilter1D(inchannels / 4, outchannels / 4, kwidth, wcval);

            QuaternionMap1D y = Reference(x, w, kwidth);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, kwidth), wval);

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch));

            QuaternionConvolution1D ope = new QuaternionConvolution1D(inwidth, inchannels, outchannels, kwidth, gradmode: false, batch);

            ope.Execute(x_tensor, w_tensor, y_tensor);

            float[] y_expect = y.ToArray();
            float[] y_actual = y_tensor.State;

            CollectionAssert.AreEqual(xval, x_tensor.State);
            CollectionAssert.AreEqual(wval, w_tensor.State);

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, ksize));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth));

            QuaternionConvolution1D ope = new QuaternionConvolution1D(inwidth, inchannels, outchannels, ksize);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/quaternion_convolution_1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static QuaternionMap1D Reference(QuaternionMap1D x, QuaternionFilter1D w, int kwidth) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, outw = inw - kwidth + 1;

            QuaternionMap1D y = new QuaternionMap1D(outchannels, outw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        for (int outch = 0; outch < outchannels; outch++) {
                            Quaternion sum = y[outch, ox, th];

                            for (int inch = 0; inch < inchannels; inch++) {
                                sum += x[inch, kx + ox, th] * w[inch, outch, kx];
                            }

                            y[outch, ox, th] = sum;
                        }
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 8, outchannels = 12, kwidth = 3, inwidth = 7, batch = 3;
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[batch * inwidth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            QuaternionMap1D x = new QuaternionMap1D(inchannels / 4, inwidth, batch, xcval);
            QuaternionFilter1D w = new QuaternionFilter1D(inchannels / 4, outchannels / 4, kwidth, wcval);

            QuaternionMap1D y = Reference(x, w, kwidth);

            float[] y_expect = {
                -4.992000000e-03f,  3.696000000e-03f,  4.896000000e-03f,  4.116000000e-03f,  -3.744000000e-03f,  2.736000000e-03f,
                3.744000000e-03f,  3.060000000e-03f,  -2.496000000e-03f,  1.776000000e-03f,  2.592000000e-03f,  2.004000000e-03f,
                -9.024000000e-03f,  7.920000000e-03f,  9.216000000e-03f,  8.244000000e-03f,  -7.008000000e-03f,  6.192000000e-03f,
                7.296000000e-03f,  6.420000000e-03f,  -4.992000000e-03f,  4.464000000e-03f,  5.376000000e-03f,  4.596000000e-03f,
                -1.305600000e-02f,  1.214400000e-02f,  1.353600000e-02f,  1.237200000e-02f,  -1.027200000e-02f,  9.648000000e-03f,
                1.084800000e-02f,  9.780000000e-03f,  -7.488000000e-03f,  7.152000000e-03f,  8.160000000e-03f,  7.188000000e-03f,
                -1.708800000e-02f,  1.636800000e-02f,  1.785600000e-02f,  1.650000000e-02f,  -1.353600000e-02f,  1.310400000e-02f,
                1.440000000e-02f,  1.314000000e-02f,  -9.984000000e-03f,  9.840000000e-03f,  1.094400000e-02f,  9.780000000e-03f,
                -2.112000000e-02f,  2.059200000e-02f,  2.217600000e-02f,  2.062800000e-02f,  -1.680000000e-02f,  1.656000000e-02f,
                1.795200000e-02f,  1.650000000e-02f,  -1.248000000e-02f,  1.252800000e-02f,  1.372800000e-02f,  1.237200000e-02f,
                -3.321600000e-02f,  3.326400000e-02f,  3.513600000e-02f,  3.301200000e-02f,  -2.659200000e-02f,  2.692800000e-02f,
                2.860800000e-02f,  2.658000000e-02f,  -1.996800000e-02f,  2.059200000e-02f,  2.208000000e-02f,  2.014800000e-02f,
                -3.724800000e-02f,  3.748800000e-02f,  3.945600000e-02f,  3.714000000e-02f,  -2.985600000e-02f,  3.038400000e-02f,
                3.216000000e-02f,  2.994000000e-02f,  -2.246400000e-02f,  2.328000000e-02f,  2.486400000e-02f,  2.274000000e-02f,
                -4.128000000e-02f,  4.171200000e-02f,  4.377600000e-02f,  4.126800000e-02f,  -3.312000000e-02f,  3.384000000e-02f,
                3.571200000e-02f,  3.330000000e-02f,  -2.496000000e-02f,  2.596800000e-02f,  2.764800000e-02f,  2.533200000e-02f,
                -4.531200000e-02f,  4.593600000e-02f,  4.809600000e-02f,  4.539600000e-02f,  -3.638400000e-02f,  3.729600000e-02f,
                3.926400000e-02f,  3.666000000e-02f,  -2.745600000e-02f,  2.865600000e-02f,  3.043200000e-02f,  2.792400000e-02f,
                -4.934400000e-02f,  5.016000000e-02f,  5.241600000e-02f,  4.952400000e-02f,  -3.964800000e-02f,  4.075200000e-02f,
                4.281600000e-02f,  4.002000000e-02f,  -2.995200000e-02f,  3.134400000e-02f,  3.321600000e-02f,  3.051600000e-02f,
                -6.144000000e-02f,  6.283200000e-02f,  6.537600000e-02f,  6.190800000e-02f,  -4.944000000e-02f,  5.112000000e-02f,
                5.347200000e-02f,  5.010000000e-02f,  -3.744000000e-02f,  3.940800000e-02f,  4.156800000e-02f,  3.829200000e-02f,
                -6.547200000e-02f,  6.705600000e-02f,  6.969600000e-02f,  6.603600000e-02f,  -5.270400000e-02f,  5.457600000e-02f,
                5.702400000e-02f,  5.346000000e-02f,  -3.993600000e-02f,  4.209600000e-02f,  4.435200000e-02f,  4.088400000e-02f,
                -6.950400000e-02f,  7.128000000e-02f,  7.401600000e-02f,  7.016400000e-02f,  -5.596800000e-02f,  5.803200000e-02f,
                6.057600000e-02f,  5.682000000e-02f,  -4.243200000e-02f,  4.478400000e-02f,  4.713600000e-02f,  4.347600000e-02f,
                -7.353600000e-02f,  7.550400000e-02f,  7.833600000e-02f,  7.429200000e-02f,  -5.923200000e-02f,  6.148800000e-02f,
                6.412800000e-02f,  6.018000000e-02f,  -4.492800000e-02f,  4.747200000e-02f,  4.992000000e-02f,  4.606800000e-02f,
                -7.756800000e-02f,  7.972800000e-02f,  8.265600000e-02f,  7.842000000e-02f,  -6.249600000e-02f,  6.494400000e-02f,
                6.768000000e-02f,  6.354000000e-02f,  -4.742400000e-02f,  5.016000000e-02f,  5.270400000e-02f,  4.866000000e-02f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");
        }
    }
}
