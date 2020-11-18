using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.QuaternionConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Quaternion {
    [TestClass]
    public class QuaternionDeconvolution1DTest {
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

                                float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                float[] wval = (new float[kwidth * inchannels * outchannels / 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                                    .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

                                Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                                    .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                QuaternionMap1D y = new QuaternionMap1D(outchannels / 4, outwidth, batch, ycval);
                                QuaternionFilter1D w = new QuaternionFilter1D(inchannels / 4, outchannels / 4, kwidth, wcval);

                                QuaternionMap1D x = Reference(y, w, inwidth, kwidth);

                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);
                                OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, kwidth), wval);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch));

                                QuaternionDeconvolution1D ope = new QuaternionDeconvolution1D(outwidth, outchannels, inchannels, kwidth, gradmode: false, batch);

                                ope.Execute(y_tensor, w_tensor, x_tensor);

                                float[] x_expect = x.ToArray();
                                float[] x_actual = x_tensor.State.Value;

                                CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                                CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

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

            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * inchannels * outchannels / 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            QuaternionMap1D y = new QuaternionMap1D(outchannels / 4, outwidth, batch, ycval);
            QuaternionFilter1D w = new QuaternionFilter1D(inchannels / 4, outchannels / 4, kwidth, wcval);

            QuaternionMap1D x = Reference(y, w, inwidth, kwidth);

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, kwidth), wval);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch));

            QuaternionDeconvolution1D ope = new QuaternionDeconvolution1D(outwidth, outchannels, inchannels, kwidth, gradmode: false, batch);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            float[] x_expect = x.ToArray();
            float[] x_actual = x_tensor.State.Value;

            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 4, ksize));

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));

            QuaternionDeconvolution1D ope = new QuaternionDeconvolution1D(outwidth, outchannels, inchannels, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/quaternion_deconvolution_1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        public static QuaternionMap1D Reference(QuaternionMap1D y, QuaternionFilter1D w, int inw, int kwidth) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int outw = inw - kwidth + 1;

            if (y.Width != outw) {
                throw new ArgumentException("mismatch shape");
            }

            QuaternionMap1D x = new QuaternionMap1D(inchannels, inw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        for (int outch = 0; outch < outchannels; outch++) {
                            Quaternion v = y[outch, ox, th];

                            for (int inch = 0; inch < inchannels; inch++) {
                                x[inch, kx + ox, th] += v * w[inch, outch, kx];
                            }
                        }
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 8, outchannels = 12, kwidth = 3, inwidth = 7, batch = 3;
            int outwidth = inwidth - kwidth + 1;

            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            QuaternionMap1D y = new QuaternionMap1D(outchannels / 4, outwidth, batch, ycval);
            QuaternionFilter1D w = new QuaternionFilter1D(inchannels / 4, outchannels / 4, kwidth, wcval);

            QuaternionMap1D x = Reference(y, w, inwidth, kwidth);

            float[] x_expect = {
                -2.404000000e-03f,  1.360000000e-03f,  2.140000000e-03f,  1.714000000e-03f,  -2.236000000e-03f,  1.264000000e-03f,
                1.996000000e-03f,  1.594000000e-03f,  -8.120000000e-03f,  6.608000000e-03f,  7.952000000e-03f,  7.100000000e-03f,
                -7.496000000e-03f,  6.128000000e-03f,  7.376000000e-03f,  6.572000000e-03f,  -1.542000000e-02f,  1.401600000e-02f,
                1.570800000e-02f,  1.443000000e-02f,  -1.405200000e-02f,  1.286400000e-02f,  1.441200000e-02f,  1.320600000e-02f,
                -2.319600000e-02f,  2.222400000e-02f,  2.413200000e-02f,  2.242200000e-02f,  -2.096400000e-02f,  2.020800000e-02f,
                2.197200000e-02f,  2.033400000e-02f,  -3.097200000e-02f,  3.043200000e-02f,  3.255600000e-02f,  3.041400000e-02f,
                -2.787600000e-02f,  2.755200000e-02f,  2.953200000e-02f,  2.746200000e-02f,  -1.474400000e-02f,  1.496000000e-02f,
                1.616000000e-02f,  1.473200000e-02f,  -1.239200000e-02f,  1.275200000e-02f,  1.385600000e-02f,  1.247600000e-02f,
                -3.844000000e-03f,  4.240000000e-03f,  4.732000000e-03f,  4.018000000e-03f,  -2.524000000e-03f,  2.992000000e-03f,
                3.436000000e-03f,  2.746000000e-03f,  -2.400400000e-02f,  2.368000000e-02f,  2.482000000e-02f,  2.367400000e-02f,
                -2.239600000e-02f,  2.214400000e-02f,  2.323600000e-02f,  2.211400000e-02f,  -4.268000000e-02f,  4.260800000e-02f,
                4.467200000e-02f,  4.238000000e-02f,  -3.917600000e-02f,  3.924800000e-02f,  4.121600000e-02f,  3.897200000e-02f,
                -5.430000000e-02f,  5.505600000e-02f,  5.782800000e-02f,  5.439000000e-02f,  -4.861200000e-02f,  4.958400000e-02f,
                5.221200000e-02f,  4.884600000e-02f,  -6.207600000e-02f,  6.326400000e-02f,  6.625200000e-02f,  6.238200000e-02f,
                -5.552400000e-02f,  5.692800000e-02f,  5.977200000e-02f,  5.597400000e-02f,  -6.985200000e-02f,  7.147200000e-02f,
                7.467600000e-02f,  7.037400000e-02f,  -6.243600000e-02f,  6.427200000e-02f,  6.733200000e-02f,  6.310200000e-02f,
                -3.202400000e-02f,  3.368000000e-02f,  3.560000000e-02f,  3.273200000e-02f,  -2.679200000e-02f,  2.859200000e-02f,
                3.041600000e-02f,  2.759600000e-02f,  -8.164000000e-03f,  9.280000000e-03f,  1.013200000e-02f,  8.698000000e-03f,
                -5.404000000e-03f,  6.592000000e-03f,  7.396000000e-03f,  5.986000000e-03f,  -4.560400000e-02f,  4.600000000e-02f,
                4.750000000e-02f,  4.563400000e-02f,  -4.255600000e-02f,  4.302400000e-02f,  4.447600000e-02f,  4.263400000e-02f,
                -7.724000000e-02f,  7.860800000e-02f,  8.139200000e-02f,  7.766000000e-02f,  -7.085600000e-02f,  7.236800000e-02f,
                7.505600000e-02f,  7.137200000e-02f,  -9.318000000e-02f,  9.609600000e-02f,  9.994800000e-02f,  9.435000000e-02f,
                -8.317200000e-02f,  8.630400000e-02f,  9.001200000e-02f,  8.448600000e-02f,  -1.009560000e-01f,  1.043040000e-01f,
                1.083720000e-01f,  1.023420000e-01f,  -9.008400000e-02f,  9.364800000e-02f,  9.757200000e-02f,  9.161400000e-02f,
                -1.087320000e-01f,  1.125120000e-01f,  1.167960000e-01f,  1.103340000e-01f,  -9.699600000e-02f,  1.009920000e-01f,
                1.051320000e-01f,  9.874200000e-02f,  -4.930400000e-02f,  5.240000000e-02f,  5.504000000e-02f,  5.073200000e-02f,
                -4.119200000e-02f,  4.443200000e-02f,  4.697600000e-02f,  4.271600000e-02f,  -1.248400000e-02f,  1.432000000e-02f,
                1.553200000e-02f,  1.337800000e-02f,  -8.284000000e-03f,  1.019200000e-02f,  1.135600000e-02f,  9.226000000e-03f,
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");
        }
    }
}
