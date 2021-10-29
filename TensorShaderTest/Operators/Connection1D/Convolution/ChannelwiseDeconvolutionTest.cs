using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class ChannelwiseDeconvolutionTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int kwidth in new int[] { 1, 3, 5 }) {
                        foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                            int outwidth = inwidth - kwidth + 1;

                            float[] yval = (new float[outwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[kwidth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map1D y = new(channels, outwidth, batch, yval);
                            Filter1D w = new(channels, 1, kwidth, wval);

                            Map1D x = Reference(y, w, inwidth, kwidth);

                            OverflowCheckedTensor y_tensor = new(Shape.Map1D(channels, outwidth, batch), yval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(channels, 1, kwidth), wval);

                            OverflowCheckedTensor x_tensor = new(Shape.Map1D(channels, inwidth, batch));

                            ChannelwiseDeconvolution ope = new(outwidth, channels, kwidth, batch);

                            ope.Execute(y_tensor, w_tensor, x_tensor);

                            float[] x_expect = x.ToArray();
                            float[] x_actual = x_tensor.State.Value;

                            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                            AssertError.Tolerance(x_expect, x_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {channels},{kwidth},{inwidth},{batch}");

                            Console.WriteLine($"pass: {channels},{kwidth},{inwidth},{batch}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void ExecuteFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int kwidth in new int[] { 1, 3, 5 }) {
                        foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                            int outwidth = inwidth - kwidth + 1;

                            float[] yval = (new float[outwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[kwidth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map1D y = new(channels, outwidth, batch, yval);
                            Filter1D w = new(channels, 1, kwidth, wval);

                            Map1D x = Reference(y, w, inwidth, kwidth);

                            OverflowCheckedTensor y_tensor = new(Shape.Map1D(channels, outwidth, batch), yval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(channels, 1, kwidth), wval);

                            OverflowCheckedTensor x_tensor = new(Shape.Map1D(channels, inwidth, batch));

                            ChannelwiseDeconvolution ope = new(outwidth, channels, kwidth, batch);

                            ope.Execute(y_tensor, w_tensor, x_tensor);

                            float[] x_expect = x.ToArray();
                            float[] x_actual = x_tensor.State.Value;

                            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{inwidth},{batch}");

                            Console.WriteLine($"pass: {channels},{kwidth},{inwidth},{batch}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void LargeMapTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            float max_err = 0;

            Random random = new(1234);

            int batch = 3;
            int channels = 49;
            int kwidth = 5;
            int inwidth = 125;
            int outwidth = inwidth - kwidth + 1;

            float[] yval = (new float[outwidth * channels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * channels]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map1D y = new(channels, outwidth, batch, yval);
            Filter1D w = new(channels, 1, kwidth, wval);

            Map1D x = Reference(y, w, inwidth, kwidth);

            OverflowCheckedTensor y_tensor = new(Shape.Map1D(channels, outwidth, batch), yval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(channels, 1, kwidth), wval);

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(channels, inwidth, batch));

            ChannelwiseDeconvolution ope = new(outwidth, channels, kwidth, batch);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            float[] x_expect = x.ToArray();
            float[] x_actual = x_tensor.State.Value;

            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"pass: {channels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 512, channels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map1D(channels, outwidth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(channels, 1, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(channels, inwidth));

            ChannelwiseDeconvolution ope = new(outwidth, channels, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/chwise_deconvolution_1d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 512, channels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map1D(channels, outwidth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(channels, 1, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(channels, inwidth));

            ChannelwiseDeconvolution ope = new(outwidth, channels, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/chwise_deconvolution_1d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map1D Reference(Map1D y, Filter1D w, int inw, int kwidth) {
            int channels = w.InChannels, batch = y.Batch;
            int outw = inw - kwidth + 1;

            if (y.Width != outw) {
                throw new ArgumentException("mismatch shape");
            }

            Map1D x = new(channels, inw, batch);

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

            Map1D y = new(channels, outwidth, batch, yval);
            Filter1D w = new(channels, 1, kwidth, wval);

            Map1D x = Reference(y, w, inwidth, kwidth);

            float[] x_expect = {
                0.0000000e+00f,  1.9000000e-05f,  3.6000000e-05f,  5.1000000e-05f,  6.4000000e-05f,  7.5000000e-05f,  8.4000000e-05f,
                1.4000000e-04f,  1.6400000e-04f,  1.8400000e-04f,  2.0000000e-04f,  2.1200000e-04f,  2.2000000e-04f,  2.2400000e-04f,
                3.7100000e-04f,  3.8600000e-04f,  3.9500000e-04f,  3.9800000e-04f,  3.9500000e-04f,  3.8600000e-04f,  3.7100000e-04f,
                6.4400000e-04f,  6.3800000e-04f,  6.2600000e-04f,  6.0800000e-04f,  5.8400000e-04f,  5.5400000e-04f,  5.1800000e-04f,
                9.1700000e-04f,  8.9000000e-04f,  8.5700000e-04f,  8.1800000e-04f,  7.7300000e-04f,  7.2200000e-04f,  6.6500000e-04f,
                1.1900000e-03f,  1.1420000e-03f,  1.0880000e-03f,  1.0280000e-03f,  9.6200000e-04f,  8.9000000e-04f,  8.1200000e-04f,
                1.4630000e-03f,  1.3940000e-03f,  1.3190000e-03f,  1.2380000e-03f,  1.1510000e-03f,  1.0580000e-03f,  9.5900000e-04f,
                1.7360000e-03f,  1.6460000e-03f,  1.5500000e-03f,  1.4480000e-03f,  1.3400000e-03f,  1.2260000e-03f,  1.1060000e-03f,
                2.0090000e-03f,  1.8980000e-03f,  1.7810000e-03f,  1.6580000e-03f,  1.5290000e-03f,  1.3940000e-03f,  1.2530000e-03f,
                2.2820000e-03f,  2.1500000e-03f,  2.0120000e-03f,  1.8680000e-03f,  1.7180000e-03f,  1.5620000e-03f,  1.4000000e-03f,
                2.5550000e-03f,  2.4020000e-03f,  2.2430000e-03f,  2.0780000e-03f,  1.9070000e-03f,  1.7300000e-03f,  1.5470000e-03f,
                1.2880000e-03f,  1.1720000e-03f,  1.0520000e-03f,  9.2800000e-04f,  8.0000000e-04f,  6.6800000e-04f,  5.3200000e-04f,
                4.2000000e-04f,  3.5500000e-04f,  2.8800000e-04f,  2.1900000e-04f,  1.4800000e-04f,  7.5000000e-05f,  0.0000000e+00f,
                1.5400000e-03f,  1.4820000e-03f,  1.4220000e-03f,  1.3600000e-03f,  1.2960000e-03f,  1.2300000e-03f,  1.1620000e-03f,
                2.6810000e-03f,  2.5510000e-03f,  2.4170000e-03f,  2.2790000e-03f,  2.1370000e-03f,  1.9910000e-03f,  1.8410000e-03f,
                3.3740000e-03f,  3.1580000e-03f,  2.9360000e-03f,  2.7080000e-03f,  2.4740000e-03f,  2.2340000e-03f,  1.9880000e-03f,
                3.6470000e-03f,  3.4100000e-03f,  3.1670000e-03f,  2.9180000e-03f,  2.6630000e-03f,  2.4020000e-03f,  2.1350000e-03f,
                3.9200000e-03f,  3.6620000e-03f,  3.3980000e-03f,  3.1280000e-03f,  2.8520000e-03f,  2.5700000e-03f,  2.2820000e-03f,
                4.1930000e-03f,  3.9140000e-03f,  3.6290000e-03f,  3.3380000e-03f,  3.0410000e-03f,  2.7380000e-03f,  2.4290000e-03f,
                4.4660000e-03f,  4.1660000e-03f,  3.8600000e-03f,  3.5480000e-03f,  3.2300000e-03f,  2.9060000e-03f,  2.5760000e-03f,
                4.7390000e-03f,  4.4180000e-03f,  4.0910000e-03f,  3.7580000e-03f,  3.4190000e-03f,  3.0740000e-03f,  2.7230000e-03f,
                5.0120000e-03f,  4.6700000e-03f,  4.3220000e-03f,  3.9680000e-03f,  3.6080000e-03f,  3.2420000e-03f,  2.8700000e-03f,
                5.2850000e-03f,  4.9220000e-03f,  4.5530000e-03f,  4.1780000e-03f,  3.7970000e-03f,  3.4100000e-03f,  3.0170000e-03f,
                5.5580000e-03f,  5.1740000e-03f,  4.7840000e-03f,  4.3880000e-03f,  3.9860000e-03f,  3.5780000e-03f,  3.1640000e-03f,
                2.7510000e-03f,  2.4810000e-03f,  2.2070000e-03f,  1.9290000e-03f,  1.6470000e-03f,  1.3610000e-03f,  1.0710000e-03f,
                8.8200000e-04f,  7.4000000e-04f,  5.9600000e-04f,  4.5000000e-04f,  3.0200000e-04f,  1.5200000e-04f,  0.0000000e+00f,
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {channels},{kwidth},{inwidth},{batch}");
        }
    }
}
