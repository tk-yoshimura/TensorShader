using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class ChannelwiseDeconvolutionTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int kheight in new int[] { 1, 3, 5 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                    int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                    float[] yval = (new float[outwidth * outheight * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * kheight * channels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                    Map2D y = new(channels, outwidth, outheight, batch, yval);
                                    Filter2D w = new(channels, 1, kwidth, kheight, wval);

                                    Map2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

                                    OverflowCheckedTensor y_tensor = new(Shape.Map2D(channels, outwidth, outheight, batch), yval);
                                    OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(channels, 1, kwidth, kheight), wval);

                                    OverflowCheckedTensor x_tensor = new(Shape.Map2D(channels, inwidth, inheight, batch));

                                    ChannelwiseDeconvolution ope = new(outwidth, outheight, channels, kwidth, kheight, batch);

                                    ope.Execute(y_tensor, w_tensor, x_tensor);

                                    float[] x_expect = x.ToArray();
                                    float[] x_actual = x_tensor.State.Value;

                                    CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                                    CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                    AssertError.Tolerance(x_expect, x_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

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
        public void ExecuteFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int kheight in new int[] { 1, 3, 5 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                    int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                    float[] yval = (new float[outwidth * outheight * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * kheight * channels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                    Map2D y = new(channels, outwidth, outheight, batch, yval);
                                    Filter2D w = new(channels, 1, kwidth, kheight, wval);

                                    Map2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

                                    OverflowCheckedTensor y_tensor = new(Shape.Map2D(channels, outwidth, outheight, batch), yval);
                                    OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(channels, 1, kwidth, kheight), wval);

                                    OverflowCheckedTensor x_tensor = new(Shape.Map2D(channels, inwidth, inheight, batch));

                                    ChannelwiseDeconvolution ope = new(outwidth, outheight, channels, kwidth, kheight, batch);

                                    ope.Execute(y_tensor, w_tensor, x_tensor);

                                    float[] x_expect = x.ToArray();
                                    float[] x_actual = x_tensor.State.Value;

                                    CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                                    CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                    AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

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
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            float max_err = 0;

            Random random = new(1234);

            int batch = 3;
            int channels = 49;
            int kwidth = 5, kheight = 3;
            int inwidth = 250, inheight = 196;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] yval = (new float[outwidth * outheight * channels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * kheight * channels]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map2D y = new(channels, outwidth, outheight, batch, yval);
            Filter2D w = new(channels, 1, kwidth, kheight, wval);

            Map2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(channels, outwidth, outheight, batch), yval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(channels, 1, kwidth, kheight), wval);

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(channels, inwidth, inheight, batch));

            ChannelwiseDeconvolution ope = new(outwidth, outheight, channels, kwidth, kheight, batch);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            float[] x_expect = x.ToArray();
            float[] x_actual = x_tensor.State.Value;

            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"pass: {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 512, inheight = 512, channels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(channels, outwidth, outheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(channels, 1, ksize, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(channels, inwidth, inheight));

            ChannelwiseDeconvolution ope = new(outwidth, outheight, channels, ksize, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/chwise_deconvolution_2d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 512, inheight = 512, channels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(channels, outwidth, outheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel2D(channels, 1, ksize, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(channels, inwidth, inheight));

            ChannelwiseDeconvolution ope = new(outwidth, outheight, channels, ksize, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/chwise_deconvolution_2d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map2D Reference(Map2D y, Filter2D w, int inw, int inh, int kwidth, int kheight) {
            int channels = w.InChannels, batch = y.Batch;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1;

            if (y.Width != outw || y.Height != outh) {
                throw new ArgumentException("mismatch shape");
            }

            Map2D x = new(channels, inw, inh, batch);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ox, oy = 0; oy < outh; oy++) {
                            for (ox = 0; ox < outw; ox++) {
                                for (int ch = 0; ch < channels; ch++) {
                                    x[ch, kx + ox, ky + oy, th] += y[ch, ox, oy, th] * w[ch, 0, kx, ky];
                                }
                            }
                        }
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int channels = 7, kwidth = 3, kheight = 5, inwidth = 13, inheight = 17, batch = 2;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] yval = (new float[outwidth * outheight * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map2D y = new(channels, outwidth, outheight, batch, yval);
            Filter2D w = new(channels, 1, kwidth, kheight, wval);

            Map2D x = Reference(y, w, inwidth, inheight, kwidth, kheight);

            float[] x_expect = {
                0.0000000e+00f,  1.0300000e-04f,  2.0400000e-04f,  3.0300000e-04f,  4.0000000e-04f,  4.9500000e-04f,  5.8800000e-04f,
                7.2800000e-04f,  9.2000000e-04f,  1.1080000e-03f,  1.2920000e-03f,  1.4720000e-03f,  1.6480000e-03f,  1.8200000e-03f,
                2.1350000e-03f,  2.4020000e-03f,  2.6630000e-03f,  2.9180000e-03f,  3.1670000e-03f,  3.4100000e-03f,  3.6470000e-03f,
                4.1720000e-03f,  4.4180000e-03f,  4.6580000e-03f,  4.8920000e-03f,  5.1200000e-03f,  5.3420000e-03f,  5.5580000e-03f,
                6.2090000e-03f,  6.4340000e-03f,  6.6530000e-03f,  6.8660000e-03f,  7.0730000e-03f,  7.2740000e-03f,  7.4690000e-03f,
                8.2460000e-03f,  8.4500000e-03f,  8.6480000e-03f,  8.8400000e-03f,  9.0260000e-03f,  9.2060000e-03f,  9.3800000e-03f,
                1.0283000e-02f,  1.0466000e-02f,  1.0643000e-02f,  1.0814000e-02f,  1.0979000e-02f,  1.1138000e-02f,  1.1291000e-02f,
                1.2320000e-02f,  1.2482000e-02f,  1.2638000e-02f,  1.2788000e-02f,  1.2932000e-02f,  1.3070000e-02f,  1.3202000e-02f,
                1.4357000e-02f,  1.4498000e-02f,  1.4633000e-02f,  1.4762000e-02f,  1.4885000e-02f,  1.5002000e-02f,  1.5113000e-02f,
                1.6394000e-02f,  1.6514000e-02f,  1.6628000e-02f,  1.6736000e-02f,  1.6838000e-02f,  1.6934000e-02f,  1.7024000e-02f,
                1.8431000e-02f,  1.8530000e-02f,  1.8623000e-02f,  1.8710000e-02f,  1.8791000e-02f,  1.8866000e-02f,  1.8935000e-02f,
                1.2460000e-02f,  1.2512000e-02f,  1.2560000e-02f,  1.2604000e-02f,  1.2644000e-02f,  1.2680000e-02f,  1.2712000e-02f,
                6.3000000e-03f,  6.3190000e-03f,  6.3360000e-03f,  6.3510000e-03f,  6.3640000e-03f,  6.3750000e-03f,  6.3840000e-03f,
                8.0080000e-03f,  8.1160000e-03f,  8.2200000e-03f,  8.3200000e-03f,  8.4160000e-03f,  8.5080000e-03f,  8.5960000e-03f,
                1.6786000e-02f,  1.6974000e-02f,  1.7154000e-02f,  1.7326000e-02f,  1.7490000e-02f,  1.7646000e-02f,  1.7794000e-02f,
                2.6236000e-02f,  2.6476000e-02f,  2.6704000e-02f,  2.6920000e-02f,  2.7124000e-02f,  2.7316000e-02f,  2.7496000e-02f,
                2.9869000e-02f,  3.0067000e-02f,  3.0253000e-02f,  3.0427000e-02f,  3.0589000e-02f,  3.0739000e-02f,  3.0877000e-02f,
                3.3502000e-02f,  3.3658000e-02f,  3.3802000e-02f,  3.3934000e-02f,  3.4054000e-02f,  3.4162000e-02f,  3.4258000e-02f,
                3.7135000e-02f,  3.7249000e-02f,  3.7351000e-02f,  3.7441000e-02f,  3.7519000e-02f,  3.7585000e-02f,  3.7639000e-02f,
                4.0768000e-02f,  4.0840000e-02f,  4.0900000e-02f,  4.0948000e-02f,  4.0984000e-02f,  4.1008000e-02f,  4.1020000e-02f,
                4.4401000e-02f,  4.4431000e-02f,  4.4449000e-02f,  4.4455000e-02f,  4.4449000e-02f,  4.4431000e-02f,  4.4401000e-02f,
                4.8034000e-02f,  4.8022000e-02f,  4.7998000e-02f,  4.7962000e-02f,  4.7914000e-02f,  4.7854000e-02f,  4.7782000e-02f,
                5.1667000e-02f,  5.1613000e-02f,  5.1547000e-02f,  5.1469000e-02f,  5.1379000e-02f,  5.1277000e-02f,  5.1163000e-02f,
                5.5300000e-02f,  5.5204000e-02f,  5.5096000e-02f,  5.4976000e-02f,  5.4844000e-02f,  5.4700000e-02f,  5.4544000e-02f,
                3.6526000e-02f,  3.6434000e-02f,  3.6334000e-02f,  3.6226000e-02f,  3.6110000e-02f,  3.5986000e-02f,  3.5854000e-02f,
                1.8060000e-02f,  1.8000000e-02f,  1.7936000e-02f,  1.7868000e-02f,  1.7796000e-02f,  1.7720000e-02f,  1.7640000e-02f,
                2.2407000e-02f,  2.2422000e-02f,  2.2431000e-02f,  2.2434000e-02f,  2.2431000e-02f,  2.2422000e-02f,  2.2407000e-02f,
                4.4940000e-02f,  4.4928000e-02f,  4.4904000e-02f,  4.4868000e-02f,  4.4820000e-02f,  4.4760000e-02f,  4.4688000e-02f,
                6.7452000e-02f,  6.7371000e-02f,  6.7272000e-02f,  6.7155000e-02f,  6.7020000e-02f,  6.6867000e-02f,  6.6696000e-02f,
                7.2240000e-02f,  7.2096000e-02f,  7.1934000e-02f,  7.1754000e-02f,  7.1556000e-02f,  7.1340000e-02f,  7.1106000e-02f,
                7.7028000e-02f,  7.6821000e-02f,  7.6596000e-02f,  7.6353000e-02f,  7.6092000e-02f,  7.5813000e-02f,  7.5516000e-02f,
                8.1816000e-02f,  8.1546000e-02f,  8.1258000e-02f,  8.0952000e-02f,  8.0628000e-02f,  8.0286000e-02f,  7.9926000e-02f,
                8.6604000e-02f,  8.6271000e-02f,  8.5920000e-02f,  8.5551000e-02f,  8.5164000e-02f,  8.4759000e-02f,  8.4336000e-02f,
                9.1392000e-02f,  9.0996000e-02f,  9.0582000e-02f,  9.0150000e-02f,  8.9700000e-02f,  8.9232000e-02f,  8.8746000e-02f,
                9.6180000e-02f,  9.5721000e-02f,  9.5244000e-02f,  9.4749000e-02f,  9.4236000e-02f,  9.3705000e-02f,  9.3156000e-02f,
                1.0096800e-01f,  1.0044600e-01f,  9.9906000e-02f,  9.9348000e-02f,  9.8772000e-02f,  9.8178000e-02f,  9.7566000e-02f,
                1.0575600e-01f,  1.0517100e-01f,  1.0456800e-01f,  1.0394700e-01f,  1.0330800e-01f,  1.0265100e-01f,  1.0197600e-01f,
                6.8964000e-02f,  6.8532000e-02f,  6.8088000e-02f,  6.7632000e-02f,  6.7164000e-02f,  6.6684000e-02f,  6.6192000e-02f,
                3.3663000e-02f,  3.3426000e-02f,  3.3183000e-02f,  3.2934000e-02f,  3.2679000e-02f,  3.2418000e-02f,  3.2151000e-02f,
                4.1580000e-02f,  4.1404000e-02f,  4.1220000e-02f,  4.1028000e-02f,  4.0828000e-02f,  4.0620000e-02f,  4.0404000e-02f,
                8.1956000e-02f,  8.1548000e-02f,  8.1124000e-02f,  8.0684000e-02f,  8.0228000e-02f,  7.9756000e-02f,  7.9268000e-02f,
                1.2093200e-01f,  1.2023600e-01f,  1.1951600e-01f,  1.1877200e-01f,  1.1800400e-01f,  1.1721200e-01f,  1.1639600e-01f,
                1.2643400e-01f,  1.2565400e-01f,  1.2485000e-01f,  1.2402200e-01f,  1.2317000e-01f,  1.2229400e-01f,  1.2139400e-01f,
                1.3193600e-01f,  1.3107200e-01f,  1.3018400e-01f,  1.2927200e-01f,  1.2833600e-01f,  1.2737600e-01f,  1.2639200e-01f,
                1.3743800e-01f,  1.3649000e-01f,  1.3551800e-01f,  1.3452200e-01f,  1.3350200e-01f,  1.3245800e-01f,  1.3139000e-01f,
                1.4294000e-01f,  1.4190800e-01f,  1.4085200e-01f,  1.3977200e-01f,  1.3866800e-01f,  1.3754000e-01f,  1.3638800e-01f,
                1.4844200e-01f,  1.4732600e-01f,  1.4618600e-01f,  1.4502200e-01f,  1.4383400e-01f,  1.4262200e-01f,  1.4138600e-01f,
                1.5394400e-01f,  1.5274400e-01f,  1.5152000e-01f,  1.5027200e-01f,  1.4900000e-01f,  1.4770400e-01f,  1.4638400e-01f,
                1.5944600e-01f,  1.5816200e-01f,  1.5685400e-01f,  1.5552200e-01f,  1.5416600e-01f,  1.5278600e-01f,  1.5138200e-01f,
                1.6494800e-01f,  1.6358000e-01f,  1.6218800e-01f,  1.6077200e-01f,  1.5933200e-01f,  1.5786800e-01f,  1.5638000e-01f,
                1.0654000e-01f,  1.0557200e-01f,  1.0458800e-01f,  1.0358800e-01f,  1.0257200e-01f,  1.0154000e-01f,  1.0049200e-01f,
                5.1492000e-02f,  5.0980000e-02f,  5.0460000e-02f,  4.9932000e-02f,  4.9396000e-02f,  4.8852000e-02f,  4.8300000e-02f,
                6.3910000e-02f,  6.3445000e-02f,  6.2970000e-02f,  6.2485000e-02f,  6.1990000e-02f,  6.1485000e-02f,  6.0970000e-02f,
                1.2460000e-01f,  1.2360000e-01f,  1.2258000e-01f,  1.2154000e-01f,  1.2048000e-01f,  1.1940000e-01f,  1.1830000e-01f,
                1.8182500e-01f,  1.8022000e-01f,  1.7858500e-01f,  1.7692000e-01f,  1.7522500e-01f,  1.7350000e-01f,  1.7174500e-01f,
                1.8760000e-01f,  1.8589000e-01f,  1.8415000e-01f,  1.8238000e-01f,  1.8058000e-01f,  1.7875000e-01f,  1.7689000e-01f,
                1.9337500e-01f,  1.9156000e-01f,  1.8971500e-01f,  1.8784000e-01f,  1.8593500e-01f,  1.8400000e-01f,  1.8203500e-01f,
                1.9915000e-01f,  1.9723000e-01f,  1.9528000e-01f,  1.9330000e-01f,  1.9129000e-01f,  1.8925000e-01f,  1.8718000e-01f,
                2.0492500e-01f,  2.0290000e-01f,  2.0084500e-01f,  1.9876000e-01f,  1.9664500e-01f,  1.9450000e-01f,  1.9232500e-01f,
                2.1070000e-01f,  2.0857000e-01f,  2.0641000e-01f,  2.0422000e-01f,  2.0200000e-01f,  1.9975000e-01f,  1.9747000e-01f,
                2.1647500e-01f,  2.1424000e-01f,  2.1197500e-01f,  2.0968000e-01f,  2.0735500e-01f,  2.0500000e-01f,  2.0261500e-01f,
                2.2225000e-01f,  2.1991000e-01f,  2.1754000e-01f,  2.1514000e-01f,  2.1271000e-01f,  2.1025000e-01f,  2.0776000e-01f,
                2.2802500e-01f,  2.2558000e-01f,  2.2310500e-01f,  2.2060000e-01f,  2.1806500e-01f,  2.1550000e-01f,  2.1290500e-01f,
                1.4602000e-01f,  1.4432000e-01f,  1.4260000e-01f,  1.4086000e-01f,  1.3910000e-01f,  1.3732000e-01f,  1.3552000e-01f,
                6.9930000e-02f,  6.9045000e-02f,  6.8150000e-02f,  6.7245000e-02f,  6.6330000e-02f,  6.5405000e-02f,  6.4470000e-02f,
                8.7780000e-02f,  8.6930000e-02f,  8.6070000e-02f,  8.5200000e-02f,  8.4320000e-02f,  8.3430000e-02f,  8.2530000e-02f,
                1.6964500e-01f,  1.6787500e-01f,  1.6608500e-01f,  1.6427500e-01f,  1.6244500e-01f,  1.6059500e-01f,  1.5872500e-01f,
                2.4535000e-01f,  2.4259000e-01f,  2.3980000e-01f,  2.3698000e-01f,  2.3413000e-01f,  2.3125000e-01f,  2.2834000e-01f,
                2.5112500e-01f,  2.4826000e-01f,  2.4536500e-01f,  2.4244000e-01f,  2.3948500e-01f,  2.3650000e-01f,  2.3348500e-01f,
                2.5690000e-01f,  2.5393000e-01f,  2.5093000e-01f,  2.4790000e-01f,  2.4484000e-01f,  2.4175000e-01f,  2.3863000e-01f,
                2.6267500e-01f,  2.5960000e-01f,  2.5649500e-01f,  2.5336000e-01f,  2.5019500e-01f,  2.4700000e-01f,  2.4377500e-01f,
                2.6845000e-01f,  2.6527000e-01f,  2.6206000e-01f,  2.5882000e-01f,  2.5555000e-01f,  2.5225000e-01f,  2.4892000e-01f,
                2.7422500e-01f,  2.7094000e-01f,  2.6762500e-01f,  2.6428000e-01f,  2.6090500e-01f,  2.5750000e-01f,  2.5406500e-01f,
                2.8000000e-01f,  2.7661000e-01f,  2.7319000e-01f,  2.6974000e-01f,  2.6626000e-01f,  2.6275000e-01f,  2.5921000e-01f,
                2.8577500e-01f,  2.8228000e-01f,  2.7875500e-01f,  2.7520000e-01f,  2.7161500e-01f,  2.6800000e-01f,  2.6435500e-01f,
                2.9155000e-01f,  2.8795000e-01f,  2.8432000e-01f,  2.8066000e-01f,  2.7697000e-01f,  2.7325000e-01f,  2.6950000e-01f,
                1.8567500e-01f,  1.8320500e-01f,  1.8071500e-01f,  1.7820500e-01f,  1.7567500e-01f,  1.7312500e-01f,  1.7055500e-01f,
                8.8410000e-02f,  8.7140000e-02f,  8.5860000e-02f,  8.4570000e-02f,  8.3270000e-02f,  8.1960000e-02f,  8.0640000e-02f,
                1.1165000e-01f,  1.1041500e-01f,  1.0917000e-01f,  1.0791500e-01f,  1.0665000e-01f,  1.0537500e-01f,  1.0409000e-01f,
                2.1469000e-01f,  2.1215000e-01f,  2.0959000e-01f,  2.0701000e-01f,  2.0441000e-01f,  2.0179000e-01f,  1.9915000e-01f,
                3.0887500e-01f,  3.0496000e-01f,  3.0101500e-01f,  2.9704000e-01f,  2.9303500e-01f,  2.8900000e-01f,  2.8493500e-01f,
                3.1465000e-01f,  3.1063000e-01f,  3.0658000e-01f,  3.0250000e-01f,  2.9839000e-01f,  2.9425000e-01f,  2.9008000e-01f,
                3.2042500e-01f,  3.1630000e-01f,  3.1214500e-01f,  3.0796000e-01f,  3.0374500e-01f,  2.9950000e-01f,  2.9522500e-01f,
                3.2620000e-01f,  3.2197000e-01f,  3.1771000e-01f,  3.1342000e-01f,  3.0910000e-01f,  3.0475000e-01f,  3.0037000e-01f,
                3.3197500e-01f,  3.2764000e-01f,  3.2327500e-01f,  3.1888000e-01f,  3.1445500e-01f,  3.1000000e-01f,  3.0551500e-01f,
                3.3775000e-01f,  3.3331000e-01f,  3.2884000e-01f,  3.2434000e-01f,  3.1981000e-01f,  3.1525000e-01f,  3.1066000e-01f,
                3.4352500e-01f,  3.3898000e-01f,  3.3440500e-01f,  3.2980000e-01f,  3.2516500e-01f,  3.2050000e-01f,  3.1580500e-01f,
                3.4930000e-01f,  3.4465000e-01f,  3.3997000e-01f,  3.3526000e-01f,  3.3052000e-01f,  3.2575000e-01f,  3.2095000e-01f,
                3.5507500e-01f,  3.5032000e-01f,  3.4553500e-01f,  3.4072000e-01f,  3.3587500e-01f,  3.3100000e-01f,  3.2609500e-01f,
                2.2533000e-01f,  2.2209000e-01f,  2.1883000e-01f,  2.1555000e-01f,  2.1225000e-01f,  2.0893000e-01f,  2.0559000e-01f,
                1.0689000e-01f,  1.0523500e-01f,  1.0357000e-01f,  1.0189500e-01f,  1.0021000e-01f,  9.8515000e-02f,  9.6810000e-02f,
                1.3552000e-01f,  1.3390000e-01f,  1.3227000e-01f,  1.3063000e-01f,  1.2898000e-01f,  1.2732000e-01f,  1.2565000e-01f,
                2.5973500e-01f,  2.5642500e-01f,  2.5309500e-01f,  2.4974500e-01f,  2.4637500e-01f,  2.4298500e-01f,  2.3957500e-01f,
                3.7240000e-01f,  3.6733000e-01f,  3.6223000e-01f,  3.5710000e-01f,  3.5194000e-01f,  3.4675000e-01f,  3.4153000e-01f,
                3.7817500e-01f,  3.7300000e-01f,  3.6779500e-01f,  3.6256000e-01f,  3.5729500e-01f,  3.5200000e-01f,  3.4667500e-01f,
                3.8395000e-01f,  3.7867000e-01f,  3.7336000e-01f,  3.6802000e-01f,  3.6265000e-01f,  3.5725000e-01f,  3.5182000e-01f,
                3.8972500e-01f,  3.8434000e-01f,  3.7892500e-01f,  3.7348000e-01f,  3.6800500e-01f,  3.6250000e-01f,  3.5696500e-01f,
                3.9550000e-01f,  3.9001000e-01f,  3.8449000e-01f,  3.7894000e-01f,  3.7336000e-01f,  3.6775000e-01f,  3.6211000e-01f,
                4.0127500e-01f,  3.9568000e-01f,  3.9005500e-01f,  3.8440000e-01f,  3.7871500e-01f,  3.7300000e-01f,  3.6725500e-01f,
                4.0705000e-01f,  4.0135000e-01f,  3.9562000e-01f,  3.8986000e-01f,  3.8407000e-01f,  3.7825000e-01f,  3.7240000e-01f,
                4.1282500e-01f,  4.0702000e-01f,  4.0118500e-01f,  3.9532000e-01f,  3.8942500e-01f,  3.8350000e-01f,  3.7754500e-01f,
                4.1860000e-01f,  4.1269000e-01f,  4.0675000e-01f,  4.0078000e-01f,  3.9478000e-01f,  3.8875000e-01f,  3.8269000e-01f,
                2.6498500e-01f,  2.6097500e-01f,  2.5694500e-01f,  2.5289500e-01f,  2.4882500e-01f,  2.4473500e-01f,  2.4062500e-01f,
                1.2537000e-01f,  1.2333000e-01f,  1.2128000e-01f,  1.1922000e-01f,  1.1715000e-01f,  1.1507000e-01f,  1.1298000e-01f,
                1.5939000e-01f,  1.5738500e-01f,  1.5537000e-01f,  1.5334500e-01f,  1.5131000e-01f,  1.4926500e-01f,  1.4721000e-01f,
                3.0478000e-01f,  3.0070000e-01f,  2.9660000e-01f,  2.9248000e-01f,  2.8834000e-01f,  2.8418000e-01f,  2.8000000e-01f,
                4.3592500e-01f,  4.2970000e-01f,  4.2344500e-01f,  4.1716000e-01f,  4.1084500e-01f,  4.0450000e-01f,  3.9812500e-01f,
                4.4170000e-01f,  4.3537000e-01f,  4.2901000e-01f,  4.2262000e-01f,  4.1620000e-01f,  4.0975000e-01f,  4.0327000e-01f,
                4.4747500e-01f,  4.4104000e-01f,  4.3457500e-01f,  4.2808000e-01f,  4.2155500e-01f,  4.1500000e-01f,  4.0841500e-01f,
                4.5325000e-01f,  4.4671000e-01f,  4.4014000e-01f,  4.3354000e-01f,  4.2691000e-01f,  4.2025000e-01f,  4.1356000e-01f,
                4.5902500e-01f,  4.5238000e-01f,  4.4570500e-01f,  4.3900000e-01f,  4.3226500e-01f,  4.2550000e-01f,  4.1870500e-01f,
                4.6480000e-01f,  4.5805000e-01f,  4.5127000e-01f,  4.4446000e-01f,  4.3762000e-01f,  4.3075000e-01f,  4.2385000e-01f,
                4.7057500e-01f,  4.6372000e-01f,  4.5683500e-01f,  4.4992000e-01f,  4.4297500e-01f,  4.3600000e-01f,  4.2899500e-01f,
                4.7635000e-01f,  4.6939000e-01f,  4.6240000e-01f,  4.5538000e-01f,  4.4833000e-01f,  4.4125000e-01f,  4.3414000e-01f,
                4.8212500e-01f,  4.7506000e-01f,  4.6796500e-01f,  4.6084000e-01f,  4.5368500e-01f,  4.4650000e-01f,  4.3928500e-01f,
                3.0464000e-01f,  2.9986000e-01f,  2.9506000e-01f,  2.9024000e-01f,  2.8540000e-01f,  2.8054000e-01f,  2.7566000e-01f,
                1.4385000e-01f,  1.4142500e-01f,  1.3899000e-01f,  1.3654500e-01f,  1.3409000e-01f,  1.3162500e-01f,  1.2915000e-01f,
                1.8326000e-01f,  1.8087000e-01f,  1.7847000e-01f,  1.7606000e-01f,  1.7364000e-01f,  1.7121000e-01f,  1.6877000e-01f,
                3.4982500e-01f,  3.4497500e-01f,  3.4010500e-01f,  3.3521500e-01f,  3.3030500e-01f,  3.2537500e-01f,  3.2042500e-01f,
                4.9945000e-01f,  4.9207000e-01f,  4.8466000e-01f,  4.7722000e-01f,  4.6975000e-01f,  4.6225000e-01f,  4.5472000e-01f,
                5.0522500e-01f,  4.9774000e-01f,  4.9022500e-01f,  4.8268000e-01f,  4.7510500e-01f,  4.6750000e-01f,  4.5986500e-01f,
                5.1100000e-01f,  5.0341000e-01f,  4.9579000e-01f,  4.8814000e-01f,  4.8046000e-01f,  4.7275000e-01f,  4.6501000e-01f,
                5.1677500e-01f,  5.0908000e-01f,  5.0135500e-01f,  4.9360000e-01f,  4.8581500e-01f,  4.7800000e-01f,  4.7015500e-01f,
                5.2255000e-01f,  5.1475000e-01f,  5.0692000e-01f,  4.9906000e-01f,  4.9117000e-01f,  4.8325000e-01f,  4.7530000e-01f,
                5.2832500e-01f,  5.2042000e-01f,  5.1248500e-01f,  5.0452000e-01f,  4.9652500e-01f,  4.8850000e-01f,  4.8044500e-01f,
                5.3410000e-01f,  5.2609000e-01f,  5.1805000e-01f,  5.0998000e-01f,  5.0188000e-01f,  4.9375000e-01f,  4.8559000e-01f,
                5.3987500e-01f,  5.3176000e-01f,  5.2361500e-01f,  5.1544000e-01f,  5.0723500e-01f,  4.9900000e-01f,  4.9073500e-01f,
                5.4565000e-01f,  5.3743000e-01f,  5.2918000e-01f,  5.2090000e-01f,  5.1259000e-01f,  5.0425000e-01f,  4.9588000e-01f,
                3.4429500e-01f,  3.3874500e-01f,  3.3317500e-01f,  3.2758500e-01f,  3.2197500e-01f,  3.1634500e-01f,  3.1069500e-01f,
                1.6233000e-01f,  1.5952000e-01f,  1.5670000e-01f,  1.5387000e-01f,  1.5103000e-01f,  1.4818000e-01f,  1.4532000e-01f,
                2.0713000e-01f,  2.0435500e-01f,  2.0157000e-01f,  1.9877500e-01f,  1.9597000e-01f,  1.9315500e-01f,  1.9033000e-01f,
                3.9487000e-01f,  3.8925000e-01f,  3.8361000e-01f,  3.7795000e-01f,  3.7227000e-01f,  3.6657000e-01f,  3.6085000e-01f,
                5.6297500e-01f,  5.5444000e-01f,  5.4587500e-01f,  5.3728000e-01f,  5.2865500e-01f,  5.2000000e-01f,  5.1131500e-01f,
                5.6875000e-01f,  5.6011000e-01f,  5.5144000e-01f,  5.4274000e-01f,  5.3401000e-01f,  5.2525000e-01f,  5.1646000e-01f,
                5.7452500e-01f,  5.6578000e-01f,  5.5700500e-01f,  5.4820000e-01f,  5.3936500e-01f,  5.3050000e-01f,  5.2160500e-01f,
                5.8030000e-01f,  5.7145000e-01f,  5.6257000e-01f,  5.5366000e-01f,  5.4472000e-01f,  5.3575000e-01f,  5.2675000e-01f,
                5.8607500e-01f,  5.7712000e-01f,  5.6813500e-01f,  5.5912000e-01f,  5.5007500e-01f,  5.4100000e-01f,  5.3189500e-01f,
                5.9185000e-01f,  5.8279000e-01f,  5.7370000e-01f,  5.6458000e-01f,  5.5543000e-01f,  5.4625000e-01f,  5.3704000e-01f,
                5.9762500e-01f,  5.8846000e-01f,  5.7926500e-01f,  5.7004000e-01f,  5.6078500e-01f,  5.5150000e-01f,  5.4218500e-01f,
                6.0340000e-01f,  5.9413000e-01f,  5.8483000e-01f,  5.7550000e-01f,  5.6614000e-01f,  5.5675000e-01f,  5.4733000e-01f,
                6.0917500e-01f,  5.9980000e-01f,  5.9039500e-01f,  5.8096000e-01f,  5.7149500e-01f,  5.6200000e-01f,  5.5247500e-01f,
                3.8395000e-01f,  3.7763000e-01f,  3.7129000e-01f,  3.6493000e-01f,  3.5855000e-01f,  3.5215000e-01f,  3.4573000e-01f,
                1.8081000e-01f,  1.7761500e-01f,  1.7441000e-01f,  1.7119500e-01f,  1.6797000e-01f,  1.6473500e-01f,  1.6149000e-01f,
                2.3100000e-01f,  2.2784000e-01f,  2.2467000e-01f,  2.2149000e-01f,  2.1830000e-01f,  2.1510000e-01f,  2.1189000e-01f,
                4.3991500e-01f,  4.3352500e-01f,  4.2711500e-01f,  4.2068500e-01f,  4.1423500e-01f,  4.0776500e-01f,  4.0127500e-01f,
                6.2650000e-01f,  6.1681000e-01f,  6.0709000e-01f,  5.9734000e-01f,  5.8756000e-01f,  5.7775000e-01f,  5.6791000e-01f,
                6.3227500e-01f,  6.2248000e-01f,  6.1265500e-01f,  6.0280000e-01f,  5.9291500e-01f,  5.8300000e-01f,  5.7305500e-01f,
                6.3805000e-01f,  6.2815000e-01f,  6.1822000e-01f,  6.0826000e-01f,  5.9827000e-01f,  5.8825000e-01f,  5.7820000e-01f,
                6.4382500e-01f,  6.3382000e-01f,  6.2378500e-01f,  6.1372000e-01f,  6.0362500e-01f,  5.9350000e-01f,  5.8334500e-01f,
                6.4960000e-01f,  6.3949000e-01f,  6.2935000e-01f,  6.1918000e-01f,  6.0898000e-01f,  5.9875000e-01f,  5.8849000e-01f,
                6.5537500e-01f,  6.4516000e-01f,  6.3491500e-01f,  6.2464000e-01f,  6.1433500e-01f,  6.0400000e-01f,  5.9363500e-01f,
                6.6115000e-01f,  6.5083000e-01f,  6.4048000e-01f,  6.3010000e-01f,  6.1969000e-01f,  6.0925000e-01f,  5.9878000e-01f,
                6.6692500e-01f,  6.5650000e-01f,  6.4604500e-01f,  6.3556000e-01f,  6.2504500e-01f,  6.1450000e-01f,  6.0392500e-01f,
                6.7270000e-01f,  6.6217000e-01f,  6.5161000e-01f,  6.4102000e-01f,  6.3040000e-01f,  6.1975000e-01f,  6.0907000e-01f,
                4.2360500e-01f,  4.1651500e-01f,  4.0940500e-01f,  4.0227500e-01f,  3.9512500e-01f,  3.8795500e-01f,  3.8076500e-01f,
                1.9929000e-01f,  1.9571000e-01f,  1.9212000e-01f,  1.8852000e-01f,  1.8491000e-01f,  1.8129000e-01f,  1.7766000e-01f,
                2.5487000e-01f,  2.5132500e-01f,  2.4777000e-01f,  2.4420500e-01f,  2.4063000e-01f,  2.3704500e-01f,  2.3345000e-01f,
                4.8496000e-01f,  4.7780000e-01f,  4.7062000e-01f,  4.6342000e-01f,  4.5620000e-01f,  4.4896000e-01f,  4.4170000e-01f,
                6.9002500e-01f,  6.7918000e-01f,  6.6830500e-01f,  6.5740000e-01f,  6.4646500e-01f,  6.3550000e-01f,  6.2450500e-01f,
                6.9580000e-01f,  6.8485000e-01f,  6.7387000e-01f,  6.6286000e-01f,  6.5182000e-01f,  6.4075000e-01f,  6.2965000e-01f,
                7.0157500e-01f,  6.9052000e-01f,  6.7943500e-01f,  6.6832000e-01f,  6.5717500e-01f,  6.4600000e-01f,  6.3479500e-01f,
                7.0735000e-01f,  6.9619000e-01f,  6.8500000e-01f,  6.7378000e-01f,  6.6253000e-01f,  6.5125000e-01f,  6.3994000e-01f,
                7.1312500e-01f,  7.0186000e-01f,  6.9056500e-01f,  6.7924000e-01f,  6.6788500e-01f,  6.5650000e-01f,  6.4508500e-01f,
                7.1890000e-01f,  7.0753000e-01f,  6.9613000e-01f,  6.8470000e-01f,  6.7324000e-01f,  6.6175000e-01f,  6.5023000e-01f,
                7.2467500e-01f,  7.1320000e-01f,  7.0169500e-01f,  6.9016000e-01f,  6.7859500e-01f,  6.6700000e-01f,  6.5537500e-01f,
                7.3045000e-01f,  7.1887000e-01f,  7.0726000e-01f,  6.9562000e-01f,  6.8395000e-01f,  6.7225000e-01f,  6.6052000e-01f,
                7.3622500e-01f,  7.2454000e-01f,  7.1282500e-01f,  7.0108000e-01f,  6.8930500e-01f,  6.7750000e-01f,  6.6566500e-01f,
                4.6326000e-01f,  4.5540000e-01f,  4.4752000e-01f,  4.3962000e-01f,  4.3170000e-01f,  4.2376000e-01f,  4.1580000e-01f,
                2.1777000e-01f,  2.1380500e-01f,  2.0983000e-01f,  2.0584500e-01f,  2.0185000e-01f,  1.9784500e-01f,  1.9383000e-01f,
                1.7463600e-01f,  1.7160400e-01f,  1.6856400e-01f,  1.6551600e-01f,  1.6246000e-01f,  1.5939600e-01f,  1.5632400e-01f,
                3.2807600e-01f,  3.2195600e-01f,  3.1582000e-01f,  3.0966800e-01f,  3.0350000e-01f,  2.9731600e-01f,  2.9111600e-01f,
                4.6012400e-01f,  4.5086000e-01f,  4.4157200e-01f,  4.3226000e-01f,  4.2292400e-01f,  4.1356400e-01f,  4.0418000e-01f,
                4.6386200e-01f,  4.5451400e-01f,  4.4514200e-01f,  4.3574600e-01f,  4.2632600e-01f,  4.1688200e-01f,  4.0741400e-01f,
                4.6760000e-01f,  4.5816800e-01f,  4.4871200e-01f,  4.3923200e-01f,  4.2972800e-01f,  4.2020000e-01f,  4.1064800e-01f,
                4.7133800e-01f,  4.6182200e-01f,  4.5228200e-01f,  4.4271800e-01f,  4.3313000e-01f,  4.2351800e-01f,  4.1388200e-01f,
                4.7507600e-01f,  4.6547600e-01f,  4.5585200e-01f,  4.4620400e-01f,  4.3653200e-01f,  4.2683600e-01f,  4.1711600e-01f,
                4.7881400e-01f,  4.6913000e-01f,  4.5942200e-01f,  4.4969000e-01f,  4.3993400e-01f,  4.3015400e-01f,  4.2035000e-01f,
                4.8255200e-01f,  4.7278400e-01f,  4.6299200e-01f,  4.5317600e-01f,  4.4333600e-01f,  4.3347200e-01f,  4.2358400e-01f,
                4.8629000e-01f,  4.7643800e-01f,  4.6656200e-01f,  4.5666200e-01f,  4.4673800e-01f,  4.3679000e-01f,  4.2681800e-01f,
                4.9002800e-01f,  4.8009200e-01f,  4.7013200e-01f,  4.6014800e-01f,  4.5014000e-01f,  4.4010800e-01f,  4.3005200e-01f,
                3.0326800e-01f,  2.9658800e-01f,  2.8989200e-01f,  2.8318000e-01f,  2.7645200e-01f,  2.6970800e-01f,  2.6294800e-01f,
                1.3986000e-01f,  1.3649200e-01f,  1.3311600e-01f,  1.2973200e-01f,  1.2634000e-01f,  1.2294000e-01f,  1.1953200e-01f,
                1.0741500e-01f,  1.0499400e-01f,  1.0256700e-01f,  1.0013400e-01f,  9.7695000e-02f,  9.5250000e-02f,  9.2799000e-02f,
                1.9790400e-01f,  1.9302000e-01f,  1.8812400e-01f,  1.8321600e-01f,  1.7829600e-01f,  1.7336400e-01f,  1.6842000e-01f,
                2.7132000e-01f,  2.6393100e-01f,  2.5652400e-01f,  2.4909900e-01f,  2.4165600e-01f,  2.3419500e-01f,  2.2671600e-01f,
                2.7346200e-01f,  2.6601000e-01f,  2.5854000e-01f,  2.5105200e-01f,  2.4354600e-01f,  2.3602200e-01f,  2.2848000e-01f,
                2.7560400e-01f,  2.6808900e-01f,  2.6055600e-01f,  2.5300500e-01f,  2.4543600e-01f,  2.3784900e-01f,  2.3024400e-01f,
                2.7774600e-01f,  2.7016800e-01f,  2.6257200e-01f,  2.5495800e-01f,  2.4732600e-01f,  2.3967600e-01f,  2.3200800e-01f,
                2.7988800e-01f,  2.7224700e-01f,  2.6458800e-01f,  2.5691100e-01f,  2.4921600e-01f,  2.4150300e-01f,  2.3377200e-01f,
                2.8203000e-01f,  2.7432600e-01f,  2.6660400e-01f,  2.5886400e-01f,  2.5110600e-01f,  2.4333000e-01f,  2.3553600e-01f,
                2.8417200e-01f,  2.7640500e-01f,  2.6862000e-01f,  2.6081700e-01f,  2.5299600e-01f,  2.4515700e-01f,  2.3730000e-01f,
                2.8631400e-01f,  2.7848400e-01f,  2.7063600e-01f,  2.6277000e-01f,  2.5488600e-01f,  2.4698400e-01f,  2.3906400e-01f,
                2.8845600e-01f,  2.8056300e-01f,  2.7265200e-01f,  2.6472300e-01f,  2.5677600e-01f,  2.4881100e-01f,  2.4082800e-01f,
                1.7371200e-01f,  1.6840800e-01f,  1.6309200e-01f,  1.5776400e-01f,  1.5242400e-01f,  1.4707200e-01f,  1.4170800e-01f,
                7.7511000e-02f,  7.4838000e-02f,  7.2159000e-02f,  6.9474000e-02f,  6.6783000e-02f,  6.4086000e-02f,  6.1383000e-02f,
                5.4824000e-02f,  5.3112000e-02f,  5.1396000e-02f,  4.9676000e-02f,  4.7952000e-02f,  4.6224000e-02f,  4.4492000e-02f,
                9.7678000e-02f,  9.4226000e-02f,  9.0766000e-02f,  8.7298000e-02f,  8.3822000e-02f,  8.0338000e-02f,  7.6846000e-02f,
                1.2846400e-01f,  1.2324400e-01f,  1.1801200e-01f,  1.1276800e-01f,  1.0751200e-01f,  1.0224400e-01f,  9.6964000e-02f,
                1.2945100e-01f,  1.2418900e-01f,  1.1891500e-01f,  1.1362900e-01f,  1.0833100e-01f,  1.0302100e-01f,  9.7699000e-02f,
                1.3043800e-01f,  1.2513400e-01f,  1.1981800e-01f,  1.1449000e-01f,  1.0915000e-01f,  1.0379800e-01f,  9.8434000e-02f,
                1.3142500e-01f,  1.2607900e-01f,  1.2072100e-01f,  1.1535100e-01f,  1.0996900e-01f,  1.0457500e-01f,  9.9169000e-02f,
                1.3241200e-01f,  1.2702400e-01f,  1.2162400e-01f,  1.1621200e-01f,  1.1078800e-01f,  1.0535200e-01f,  9.9904000e-02f,
                1.3339900e-01f,  1.2796900e-01f,  1.2252700e-01f,  1.1707300e-01f,  1.1160700e-01f,  1.0612900e-01f,  1.0063900e-01f,
                1.3438600e-01f,  1.2891400e-01f,  1.2343000e-01f,  1.1793400e-01f,  1.1242600e-01f,  1.0690600e-01f,  1.0137400e-01f,
                1.3537300e-01f,  1.2985900e-01f,  1.2433300e-01f,  1.1879500e-01f,  1.1324500e-01f,  1.0768300e-01f,  1.0210900e-01f,
                1.3636000e-01f,  1.3080400e-01f,  1.2523600e-01f,  1.1965600e-01f,  1.1406400e-01f,  1.0846000e-01f,  1.0284400e-01f,
                7.7826000e-02f,  7.4094000e-02f,  7.0354000e-02f,  6.6606000e-02f,  6.2850000e-02f,  5.9086000e-02f,  5.5314000e-02f,
                3.2340000e-02f,  3.0460000e-02f,  2.8576000e-02f,  2.6688000e-02f,  2.4796000e-02f,  2.2900000e-02f,  2.1000000e-02f,
                1.8480000e-02f,  1.7575000e-02f,  1.6668000e-02f,  1.5759000e-02f,  1.4848000e-02f,  1.3935000e-02f,  1.3020000e-02f,
                3.0632000e-02f,  2.8808000e-02f,  2.6980000e-02f,  2.5148000e-02f,  2.3312000e-02f,  2.1472000e-02f,  1.9628000e-02f,
                3.6407000e-02f,  3.3650000e-02f,  3.0887000e-02f,  2.8118000e-02f,  2.5343000e-02f,  2.2562000e-02f,  1.9775000e-02f,
                3.6680000e-02f,  3.3902000e-02f,  3.1118000e-02f,  2.8328000e-02f,  2.5532000e-02f,  2.2730000e-02f,  1.9922000e-02f,
                3.6953000e-02f,  3.4154000e-02f,  3.1349000e-02f,  2.8538000e-02f,  2.5721000e-02f,  2.2898000e-02f,  2.0069000e-02f,
                3.7226000e-02f,  3.4406000e-02f,  3.1580000e-02f,  2.8748000e-02f,  2.5910000e-02f,  2.3066000e-02f,  2.0216000e-02f,
                3.7499000e-02f,  3.4658000e-02f,  3.1811000e-02f,  2.8958000e-02f,  2.6099000e-02f,  2.3234000e-02f,  2.0363000e-02f,
                3.7772000e-02f,  3.4910000e-02f,  3.2042000e-02f,  2.9168000e-02f,  2.6288000e-02f,  2.3402000e-02f,  2.0510000e-02f,
                3.8045000e-02f,  3.5162000e-02f,  3.2273000e-02f,  2.9378000e-02f,  2.6477000e-02f,  2.3570000e-02f,  2.0657000e-02f,
                3.8318000e-02f,  3.5414000e-02f,  3.2504000e-02f,  2.9588000e-02f,  2.6666000e-02f,  2.3738000e-02f,  2.0804000e-02f,
                3.8591000e-02f,  3.5666000e-02f,  3.2735000e-02f,  2.9798000e-02f,  2.6855000e-02f,  2.3906000e-02f,  2.0951000e-02f,
                1.8844000e-02f,  1.6880000e-02f,  1.4912000e-02f,  1.2940000e-02f,  1.0964000e-02f,  8.9840000e-03f,  7.0000000e-03f,
                5.9640000e-03f,  4.9750000e-03f,  3.9840000e-03f,  2.9910000e-03f,  1.9960000e-03f,  9.9900000e-04f,  0.0000000e+00f,
                1.0410400e-01f,  1.0320600e-01f,  1.0230600e-01f,  1.0140400e-01f,  1.0050000e-01f,  9.9594000e-02f,  9.8686000e-02f,
                2.0192900e-01f,  2.0011900e-01f,  1.9830500e-01f,  1.9648700e-01f,  1.9466500e-01f,  1.9283900e-01f,  1.9100900e-01f,
                2.9342600e-01f,  2.9069000e-01f,  2.8794800e-01f,  2.8520000e-01f,  2.8244600e-01f,  2.7968600e-01f,  2.7692000e-01f,
                2.9546300e-01f,  2.9270600e-01f,  2.8994300e-01f,  2.8717400e-01f,  2.8439900e-01f,  2.8161800e-01f,  2.7883100e-01f,
                2.9750000e-01f,  2.9472200e-01f,  2.9193800e-01f,  2.8914800e-01f,  2.8635200e-01f,  2.8355000e-01f,  2.8074200e-01f,
                2.9953700e-01f,  2.9673800e-01f,  2.9393300e-01f,  2.9112200e-01f,  2.8830500e-01f,  2.8548200e-01f,  2.8265300e-01f,
                3.0157400e-01f,  2.9875400e-01f,  2.9592800e-01f,  2.9309600e-01f,  2.9025800e-01f,  2.8741400e-01f,  2.8456400e-01f,
                3.0361100e-01f,  3.0077000e-01f,  2.9792300e-01f,  2.9507000e-01f,  2.9221100e-01f,  2.8934600e-01f,  2.8647500e-01f,
                3.0564800e-01f,  3.0278600e-01f,  2.9991800e-01f,  2.9704400e-01f,  2.9416400e-01f,  2.9127800e-01f,  2.8838600e-01f,
                3.0768500e-01f,  3.0480200e-01f,  3.0191300e-01f,  2.9901800e-01f,  2.9611700e-01f,  2.9321000e-01f,  2.9029700e-01f,
                3.0972200e-01f,  3.0681800e-01f,  3.0390800e-01f,  3.0099200e-01f,  2.9807000e-01f,  2.9514200e-01f,  2.9220800e-01f,
                1.9964700e-01f,  1.9769700e-01f,  1.9574300e-01f,  1.9378500e-01f,  1.9182300e-01f,  1.8985700e-01f,  1.8788700e-01f,
                9.6390000e-02f,  9.5408000e-02f,  9.4424000e-02f,  9.3438000e-02f,  9.2450000e-02f,  9.1460000e-02f,  9.0468000e-02f,
                1.9519500e-01f,  1.9330100e-01f,  1.9140300e-01f,  1.8950100e-01f,  1.8759500e-01f,  1.8568500e-01f,  1.8377100e-01f,
                3.7714600e-01f,  3.7333000e-01f,  3.6950600e-01f,  3.6567400e-01f,  3.6183400e-01f,  3.5798600e-01f,  3.5413000e-01f,
                5.4575500e-01f,  5.3998900e-01f,  5.3421100e-01f,  5.2842100e-01f,  5.2261900e-01f,  5.1680500e-01f,  5.1097900e-01f,
                5.4938800e-01f,  5.4358000e-01f,  5.3776000e-01f,  5.3192800e-01f,  5.2608400e-01f,  5.2022800e-01f,  5.1436000e-01f,
                5.5302100e-01f,  5.4717100e-01f,  5.4130900e-01f,  5.3543500e-01f,  5.2954900e-01f,  5.2365100e-01f,  5.1774100e-01f,
                5.5665400e-01f,  5.5076200e-01f,  5.4485800e-01f,  5.3894200e-01f,  5.3301400e-01f,  5.2707400e-01f,  5.2112200e-01f,
                5.6028700e-01f,  5.5435300e-01f,  5.4840700e-01f,  5.4244900e-01f,  5.3647900e-01f,  5.3049700e-01f,  5.2450300e-01f,
                5.6392000e-01f,  5.5794400e-01f,  5.5195600e-01f,  5.4595600e-01f,  5.3994400e-01f,  5.3392000e-01f,  5.2788400e-01f,
                5.6755300e-01f,  5.6153500e-01f,  5.5550500e-01f,  5.4946300e-01f,  5.4340900e-01f,  5.3734300e-01f,  5.3126500e-01f,
                5.7118600e-01f,  5.6512600e-01f,  5.5905400e-01f,  5.5297000e-01f,  5.4687400e-01f,  5.4076600e-01f,  5.3464600e-01f,
                5.7481900e-01f,  5.6871700e-01f,  5.6260300e-01f,  5.5647700e-01f,  5.5033900e-01f,  5.4418900e-01f,  5.3802700e-01f,
                3.6885800e-01f,  3.6476200e-01f,  3.6065800e-01f,  3.5654600e-01f,  3.5242600e-01f,  3.4829800e-01f,  3.4416200e-01f,
                1.7721900e-01f,  1.7515700e-01f,  1.7309100e-01f,  1.7102100e-01f,  1.6894700e-01f,  1.6686900e-01f,  1.6478700e-01f,
                2.7165600e-01f,  2.6866800e-01f,  2.6567400e-01f,  2.6267400e-01f,  2.5966800e-01f,  2.5665600e-01f,  2.5363800e-01f,
                5.2241700e-01f,  5.1639900e-01f,  5.1036900e-01f,  5.0432700e-01f,  4.9827300e-01f,  4.9220700e-01f,  4.8612900e-01f,
                7.5213600e-01f,  7.4304600e-01f,  7.3393800e-01f,  7.2481200e-01f,  7.1566800e-01f,  7.0650600e-01f,  6.9732600e-01f,
                7.5692400e-01f,  7.4777100e-01f,  7.3860000e-01f,  7.2941100e-01f,  7.2020400e-01f,  7.1097900e-01f,  7.0173600e-01f,
                7.6171200e-01f,  7.5249600e-01f,  7.4326200e-01f,  7.3401000e-01f,  7.2474000e-01f,  7.1545200e-01f,  7.0614600e-01f,
                7.6650000e-01f,  7.5722100e-01f,  7.4792400e-01f,  7.3860900e-01f,  7.2927600e-01f,  7.1992500e-01f,  7.1055600e-01f,
                7.7128800e-01f,  7.6194600e-01f,  7.5258600e-01f,  7.4320800e-01f,  7.3381200e-01f,  7.2439800e-01f,  7.1496600e-01f,
                7.7607600e-01f,  7.6667100e-01f,  7.5724800e-01f,  7.4780700e-01f,  7.3834800e-01f,  7.2887100e-01f,  7.1937600e-01f,
                7.8086400e-01f,  7.7139600e-01f,  7.6191000e-01f,  7.5240600e-01f,  7.4288400e-01f,  7.3334400e-01f,  7.2378600e-01f,
                7.8565200e-01f,  7.7612100e-01f,  7.6657200e-01f,  7.5700500e-01f,  7.4742000e-01f,  7.3781700e-01f,  7.2819600e-01f,
                7.9044000e-01f,  7.8084600e-01f,  7.7123400e-01f,  7.6160400e-01f,  7.5195600e-01f,  7.4229000e-01f,  7.3260600e-01f,
                5.0439900e-01f,  4.9796100e-01f,  4.9151100e-01f,  4.8504900e-01f,  4.7857500e-01f,  4.7208900e-01f,  4.6559100e-01f,
                2.4087000e-01f,  2.3763000e-01f,  2.3438400e-01f,  2.3113200e-01f,  2.2787400e-01f,  2.2461000e-01f,  2.2134000e-01f,
                3.3187000e-01f,  3.2769000e-01f,  3.2350200e-01f,  3.1930600e-01f,  3.1510200e-01f,  3.1089000e-01f,  3.0667000e-01f,
                6.3450800e-01f,  6.2609200e-01f,  6.1766000e-01f,  6.0921200e-01f,  6.0074800e-01f,  5.9226800e-01f,  5.8377200e-01f,
                9.0771800e-01f,  8.9501000e-01f,  8.8227800e-01f,  8.6952200e-01f,  8.5674200e-01f,  8.4393800e-01f,  8.3111000e-01f,
                9.1322000e-01f,  9.0042800e-01f,  8.8761200e-01f,  8.7477200e-01f,  8.6190800e-01f,  8.4902000e-01f,  8.3610800e-01f,
                9.1872200e-01f,  9.0584600e-01f,  8.9294600e-01f,  8.8002200e-01f,  8.6707400e-01f,  8.5410200e-01f,  8.4110600e-01f,
                9.2422400e-01f,  9.1126400e-01f,  8.9828000e-01f,  8.8527200e-01f,  8.7224000e-01f,  8.5918400e-01f,  8.4610400e-01f,
                9.2972600e-01f,  9.1668200e-01f,  9.0361400e-01f,  8.9052200e-01f,  8.7740600e-01f,  8.6426600e-01f,  8.5110200e-01f,
                9.3522800e-01f,  9.2210000e-01f,  9.0894800e-01f,  8.9577200e-01f,  8.8257200e-01f,  8.6934800e-01f,  8.5610000e-01f,
                9.4073000e-01f,  9.2751800e-01f,  9.1428200e-01f,  9.0102200e-01f,  8.8773800e-01f,  8.7443000e-01f,  8.6109800e-01f,
                9.4623200e-01f,  9.3293600e-01f,  9.1961600e-01f,  9.0627200e-01f,  8.9290400e-01f,  8.7951200e-01f,  8.6609600e-01f,
                9.5173400e-01f,  9.3835400e-01f,  9.2495000e-01f,  9.1152200e-01f,  8.9807000e-01f,  8.8459400e-01f,  8.7109400e-01f,
                6.0303600e-01f,  5.9406000e-01f,  5.8506800e-01f,  5.7606000e-01f,  5.6703600e-01f,  5.5799600e-01f,  5.4894000e-01f,
                2.8572600e-01f,  2.8121000e-01f,  2.7668600e-01f,  2.7215400e-01f,  2.6761400e-01f,  2.6306600e-01f,  2.5851000e-01f,
                3.7422000e-01f,  3.6875000e-01f,  3.6327000e-01f,  3.5778000e-01f,  3.5228000e-01f,  3.4677000e-01f,  3.4125000e-01f,
                7.1018500e-01f,  6.9917500e-01f,  6.8814500e-01f,  6.7709500e-01f,  6.6602500e-01f,  6.5493500e-01f,  6.4382500e-01f,
                1.0076500e+00f,  9.9103000e-01f,  9.7438000e-01f,  9.5770000e-01f,  9.4099000e-01f,  9.2425000e-01f,  9.0748000e-01f,
                1.0134250e+00f,  9.9670000e-01f,  9.7994500e-01f,  9.6316000e-01f,  9.4634500e-01f,  9.2950000e-01f,  9.1262500e-01f,
                1.0192000e+00f,  1.0023700e+00f,  9.8551000e-01f,  9.6862000e-01f,  9.5170000e-01f,  9.3475000e-01f,  9.1777000e-01f,
                1.0249750e+00f,  1.0080400e+00f,  9.9107500e-01f,  9.7408000e-01f,  9.5705500e-01f,  9.4000000e-01f,  9.2291500e-01f,
                1.0307500e+00f,  1.0137100e+00f,  9.9664000e-01f,  9.7954000e-01f,  9.6241000e-01f,  9.4525000e-01f,  9.2806000e-01f,
                1.0365250e+00f,  1.0193800e+00f,  1.0022050e+00f,  9.8500000e-01f,  9.6776500e-01f,  9.5050000e-01f,  9.3320500e-01f,
                1.0423000e+00f,  1.0250500e+00f,  1.0077700e+00f,  9.9046000e-01f,  9.7312000e-01f,  9.5575000e-01f,  9.3835000e-01f,
                1.0480750e+00f,  1.0307200e+00f,  1.0133350e+00f,  9.9592000e-01f,  9.7847500e-01f,  9.6100000e-01f,  9.4349500e-01f,
                1.0538500e+00f,  1.0363900e+00f,  1.0189000e+00f,  1.0013800e+00f,  9.8383000e-01f,  9.6625000e-01f,  9.4864000e-01f,
                6.6153500e-01f,  6.4982500e-01f,  6.3809500e-01f,  6.2634500e-01f,  6.1457500e-01f,  6.0278500e-01f,  5.9097500e-01f,
                3.1017000e-01f,  3.0428000e-01f,  2.9838000e-01f,  2.9247000e-01f,  2.8655000e-01f,  2.8062000e-01f,  2.7468000e-01f,
                3.9809000e-01f,  3.9223500e-01f,  3.8637000e-01f,  3.8049500e-01f,  3.7461000e-01f,  3.6871500e-01f,  3.6281000e-01f,
                7.5523000e-01f,  7.4345000e-01f,  7.3165000e-01f,  7.1983000e-01f,  7.0799000e-01f,  6.9613000e-01f,  6.8425000e-01f,
                1.0711750e+00f,  1.0534000e+00f,  1.0355950e+00f,  1.0177600e+00f,  9.9989500e-01f,  9.8200000e-01f,  9.6407500e-01f,
                1.0769500e+00f,  1.0590700e+00f,  1.0411600e+00f,  1.0232200e+00f,  1.0052500e+00f,  9.8725000e-01f,  9.6922000e-01f,
                1.0827250e+00f,  1.0647400e+00f,  1.0467250e+00f,  1.0286800e+00f,  1.0106050e+00f,  9.9250000e-01f,  9.7436500e-01f,
                1.0885000e+00f,  1.0704100e+00f,  1.0522900e+00f,  1.0341400e+00f,  1.0159600e+00f,  9.9775000e-01f,  9.7951000e-01f,
                1.0942750e+00f,  1.0760800e+00f,  1.0578550e+00f,  1.0396000e+00f,  1.0213150e+00f,  1.0030000e+00f,  9.8465500e-01f,
                1.1000500e+00f,  1.0817500e+00f,  1.0634200e+00f,  1.0450600e+00f,  1.0266700e+00f,  1.0082500e+00f,  9.8980000e-01f,
                1.1058250e+00f,  1.0874200e+00f,  1.0689850e+00f,  1.0505200e+00f,  1.0320250e+00f,  1.0135000e+00f,  9.9494500e-01f,
                1.1116000e+00f,  1.0930900e+00f,  1.0745500e+00f,  1.0559800e+00f,  1.0373800e+00f,  1.0187500e+00f,  1.0000900e+00f,
                1.1173750e+00f,  1.0987600e+00f,  1.0801150e+00f,  1.0614400e+00f,  1.0427350e+00f,  1.0240000e+00f,  1.0052350e+00f,
                7.0119000e-01f,  6.8871000e-01f,  6.7621000e-01f,  6.6369000e-01f,  6.5115000e-01f,  6.3859000e-01f,  6.2601000e-01f,
                3.2865000e-01f,  3.2237500e-01f,  3.1609000e-01f,  3.0979500e-01f,  3.0349000e-01f,  2.9717500e-01f,  2.9085000e-01f,
                4.2196000e-01f,  4.1572000e-01f,  4.0947000e-01f,  4.0321000e-01f,  3.9694000e-01f,  3.9066000e-01f,  3.8437000e-01f,
                8.0027500e-01f,  7.8772500e-01f,  7.7515500e-01f,  7.6256500e-01f,  7.4995500e-01f,  7.3732500e-01f,  7.2467500e-01f,
                1.1347000e+00f,  1.1157700e+00f,  1.0968100e+00f,  1.0778200e+00f,  1.0588000e+00f,  1.0397500e+00f,  1.0206700e+00f,
                1.1404750e+00f,  1.1214400e+00f,  1.1023750e+00f,  1.0832800e+00f,  1.0641550e+00f,  1.0450000e+00f,  1.0258150e+00f,
                1.1462500e+00f,  1.1271100e+00f,  1.1079400e+00f,  1.0887400e+00f,  1.0695100e+00f,  1.0502500e+00f,  1.0309600e+00f,
                1.1520250e+00f,  1.1327800e+00f,  1.1135050e+00f,  1.0942000e+00f,  1.0748650e+00f,  1.0555000e+00f,  1.0361050e+00f,
                1.1578000e+00f,  1.1384500e+00f,  1.1190700e+00f,  1.0996600e+00f,  1.0802200e+00f,  1.0607500e+00f,  1.0412500e+00f,
                1.1635750e+00f,  1.1441200e+00f,  1.1246350e+00f,  1.1051200e+00f,  1.0855750e+00f,  1.0660000e+00f,  1.0463950e+00f,
                1.1693500e+00f,  1.1497900e+00f,  1.1302000e+00f,  1.1105800e+00f,  1.0909300e+00f,  1.0712500e+00f,  1.0515400e+00f,
                1.1751250e+00f,  1.1554600e+00f,  1.1357650e+00f,  1.1160400e+00f,  1.0962850e+00f,  1.0765000e+00f,  1.0566850e+00f,
                1.1809000e+00f,  1.1611300e+00f,  1.1413300e+00f,  1.1215000e+00f,  1.1016400e+00f,  1.0817500e+00f,  1.0618300e+00f,
                7.4084500e-01f,  7.2759500e-01f,  7.1432500e-01f,  7.0103500e-01f,  6.8772500e-01f,  6.7439500e-01f,  6.6104500e-01f,
                3.4713000e-01f,  3.4047000e-01f,  3.3380000e-01f,  3.2712000e-01f,  3.2043000e-01f,  3.1373000e-01f,  3.0702000e-01f,
                4.4583000e-01f,  4.3920500e-01f,  4.3257000e-01f,  4.2592500e-01f,  4.1927000e-01f,  4.1260500e-01f,  4.0593000e-01f,
                8.4532000e-01f,  8.3200000e-01f,  8.1866000e-01f,  8.0530000e-01f,  7.9192000e-01f,  7.7852000e-01f,  7.6510000e-01f,
                1.1982250e+00f,  1.1781400e+00f,  1.1580250e+00f,  1.1378800e+00f,  1.1177050e+00f,  1.0975000e+00f,  1.0772650e+00f,
                1.2040000e+00f,  1.1838100e+00f,  1.1635900e+00f,  1.1433400e+00f,  1.1230600e+00f,  1.1027500e+00f,  1.0824100e+00f,
                1.2097750e+00f,  1.1894800e+00f,  1.1691550e+00f,  1.1488000e+00f,  1.1284150e+00f,  1.1080000e+00f,  1.0875550e+00f,
                1.2155500e+00f,  1.1951500e+00f,  1.1747200e+00f,  1.1542600e+00f,  1.1337700e+00f,  1.1132500e+00f,  1.0927000e+00f,
                1.2213250e+00f,  1.2008200e+00f,  1.1802850e+00f,  1.1597200e+00f,  1.1391250e+00f,  1.1185000e+00f,  1.0978450e+00f,
                1.2271000e+00f,  1.2064900e+00f,  1.1858500e+00f,  1.1651800e+00f,  1.1444800e+00f,  1.1237500e+00f,  1.1029900e+00f,
                1.2328750e+00f,  1.2121600e+00f,  1.1914150e+00f,  1.1706400e+00f,  1.1498350e+00f,  1.1290000e+00f,  1.1081350e+00f,
                1.2386500e+00f,  1.2178300e+00f,  1.1969800e+00f,  1.1761000e+00f,  1.1551900e+00f,  1.1342500e+00f,  1.1132800e+00f,
                1.2444250e+00f,  1.2235000e+00f,  1.2025450e+00f,  1.1815600e+00f,  1.1605450e+00f,  1.1395000e+00f,  1.1184250e+00f,
                7.8050000e-01f,  7.6648000e-01f,  7.5244000e-01f,  7.3838000e-01f,  7.2430000e-01f,  7.1020000e-01f,  6.9608000e-01f,
                3.6561000e-01f,  3.5856500e-01f,  3.5151000e-01f,  3.4444500e-01f,  3.3737000e-01f,  3.3028500e-01f,  3.2319000e-01f,
                4.6970000e-01f,  4.6269000e-01f,  4.5567000e-01f,  4.4864000e-01f,  4.4160000e-01f,  4.3455000e-01f,  4.2749000e-01f,
                8.9036500e-01f,  8.7627500e-01f,  8.6216500e-01f,  8.4803500e-01f,  8.3388500e-01f,  8.1971500e-01f,  8.0552500e-01f,
                1.2617500e+00f,  1.2405100e+00f,  1.2192400e+00f,  1.1979400e+00f,  1.1766100e+00f,  1.1552500e+00f,  1.1338600e+00f,
                1.2675250e+00f,  1.2461800e+00f,  1.2248050e+00f,  1.2034000e+00f,  1.1819650e+00f,  1.1605000e+00f,  1.1390050e+00f,
                1.2733000e+00f,  1.2518500e+00f,  1.2303700e+00f,  1.2088600e+00f,  1.1873200e+00f,  1.1657500e+00f,  1.1441500e+00f,
                1.2790750e+00f,  1.2575200e+00f,  1.2359350e+00f,  1.2143200e+00f,  1.1926750e+00f,  1.1710000e+00f,  1.1492950e+00f,
                1.2848500e+00f,  1.2631900e+00f,  1.2415000e+00f,  1.2197800e+00f,  1.1980300e+00f,  1.1762500e+00f,  1.1544400e+00f,
                1.2906250e+00f,  1.2688600e+00f,  1.2470650e+00f,  1.2252400e+00f,  1.2033850e+00f,  1.1815000e+00f,  1.1595850e+00f,
                1.2964000e+00f,  1.2745300e+00f,  1.2526300e+00f,  1.2307000e+00f,  1.2087400e+00f,  1.1867500e+00f,  1.1647300e+00f,
                1.3021750e+00f,  1.2802000e+00f,  1.2581950e+00f,  1.2361600e+00f,  1.2140950e+00f,  1.1920000e+00f,  1.1698750e+00f,
                1.3079500e+00f,  1.2858700e+00f,  1.2637600e+00f,  1.2416200e+00f,  1.2194500e+00f,  1.1972500e+00f,  1.1750200e+00f,
                8.2015500e-01f,  8.0536500e-01f,  7.9055500e-01f,  7.7572500e-01f,  7.6087500e-01f,  7.4600500e-01f,  7.3111500e-01f,
                3.8409000e-01f,  3.7666000e-01f,  3.6922000e-01f,  3.6177000e-01f,  3.5431000e-01f,  3.4684000e-01f,  3.3936000e-01f,
                4.9357000e-01f,  4.8617500e-01f,  4.7877000e-01f,  4.7135500e-01f,  4.6393000e-01f,  4.5649500e-01f,  4.4905000e-01f,
                9.3541000e-01f,  9.2055000e-01f,  9.0567000e-01f,  8.9077000e-01f,  8.7585000e-01f,  8.6091000e-01f,  8.4595000e-01f,
                1.3252750e+00f,  1.3028800e+00f,  1.2804550e+00f,  1.2580000e+00f,  1.2355150e+00f,  1.2130000e+00f,  1.1904550e+00f,
                1.3310500e+00f,  1.3085500e+00f,  1.2860200e+00f,  1.2634600e+00f,  1.2408700e+00f,  1.2182500e+00f,  1.1956000e+00f,
                1.3368250e+00f,  1.3142200e+00f,  1.2915850e+00f,  1.2689200e+00f,  1.2462250e+00f,  1.2235000e+00f,  1.2007450e+00f,
                1.3426000e+00f,  1.3198900e+00f,  1.2971500e+00f,  1.2743800e+00f,  1.2515800e+00f,  1.2287500e+00f,  1.2058900e+00f,
                1.3483750e+00f,  1.3255600e+00f,  1.3027150e+00f,  1.2798400e+00f,  1.2569350e+00f,  1.2340000e+00f,  1.2110350e+00f,
                1.3541500e+00f,  1.3312300e+00f,  1.3082800e+00f,  1.2853000e+00f,  1.2622900e+00f,  1.2392500e+00f,  1.2161800e+00f,
                1.3599250e+00f,  1.3369000e+00f,  1.3138450e+00f,  1.2907600e+00f,  1.2676450e+00f,  1.2445000e+00f,  1.2213250e+00f,
                1.3657000e+00f,  1.3425700e+00f,  1.3194100e+00f,  1.2962200e+00f,  1.2730000e+00f,  1.2497500e+00f,  1.2264700e+00f,
                1.3714750e+00f,  1.3482400e+00f,  1.3249750e+00f,  1.3016800e+00f,  1.2783550e+00f,  1.2550000e+00f,  1.2316150e+00f,
                8.5981000e-01f,  8.4425000e-01f,  8.2867000e-01f,  8.1307000e-01f,  7.9745000e-01f,  7.8181000e-01f,  7.6615000e-01f,
                4.0257000e-01f,  3.9475500e-01f,  3.8693000e-01f,  3.7909500e-01f,  3.7125000e-01f,  3.6339500e-01f,  3.5553000e-01f,
                5.1744000e-01f,  5.0966000e-01f,  5.0187000e-01f,  4.9407000e-01f,  4.8626000e-01f,  4.7844000e-01f,  4.7061000e-01f,
                9.8045500e-01f,  9.6482500e-01f,  9.4917500e-01f,  9.3350500e-01f,  9.1781500e-01f,  9.0210500e-01f,  8.8637500e-01f,
                1.3888000e+00f,  1.3652500e+00f,  1.3416700e+00f,  1.3180600e+00f,  1.2944200e+00f,  1.2707500e+00f,  1.2470500e+00f,
                1.3945750e+00f,  1.3709200e+00f,  1.3472350e+00f,  1.3235200e+00f,  1.2997750e+00f,  1.2760000e+00f,  1.2521950e+00f,
                1.4003500e+00f,  1.3765900e+00f,  1.3528000e+00f,  1.3289800e+00f,  1.3051300e+00f,  1.2812500e+00f,  1.2573400e+00f,
                1.4061250e+00f,  1.3822600e+00f,  1.3583650e+00f,  1.3344400e+00f,  1.3104850e+00f,  1.2865000e+00f,  1.2624850e+00f,
                1.4119000e+00f,  1.3879300e+00f,  1.3639300e+00f,  1.3399000e+00f,  1.3158400e+00f,  1.2917500e+00f,  1.2676300e+00f,
                1.4176750e+00f,  1.3936000e+00f,  1.3694950e+00f,  1.3453600e+00f,  1.3211950e+00f,  1.2970000e+00f,  1.2727750e+00f,
                1.4234500e+00f,  1.3992700e+00f,  1.3750600e+00f,  1.3508200e+00f,  1.3265500e+00f,  1.3022500e+00f,  1.2779200e+00f,
                1.4292250e+00f,  1.4049400e+00f,  1.3806250e+00f,  1.3562800e+00f,  1.3319050e+00f,  1.3075000e+00f,  1.2830650e+00f,
                1.4350000e+00f,  1.4106100e+00f,  1.3861900e+00f,  1.3617400e+00f,  1.3372600e+00f,  1.3127500e+00f,  1.2882100e+00f,
                8.9946500e-01f,  8.8313500e-01f,  8.6678500e-01f,  8.5041500e-01f,  8.3402500e-01f,  8.1761500e-01f,  8.0118500e-01f,
                4.2105000e-01f,  4.1285000e-01f,  4.0464000e-01f,  3.9642000e-01f,  3.8819000e-01f,  3.7995000e-01f,  3.7170000e-01f,
                5.4131000e-01f,  5.3314500e-01f,  5.2497000e-01f,  5.1678500e-01f,  5.0859000e-01f,  5.0038500e-01f,  4.9217000e-01f,
                1.0255000e+00f,  1.0091000e+00f,  9.9268000e-01f,  9.7624000e-01f,  9.5978000e-01f,  9.4330000e-01f,  9.2680000e-01f,
                1.4523250e+00f,  1.4276200e+00f,  1.4028850e+00f,  1.3781200e+00f,  1.3533250e+00f,  1.3285000e+00f,  1.3036450e+00f,
                1.4581000e+00f,  1.4332900e+00f,  1.4084500e+00f,  1.3835800e+00f,  1.3586800e+00f,  1.3337500e+00f,  1.3087900e+00f,
                1.4638750e+00f,  1.4389600e+00f,  1.4140150e+00f,  1.3890400e+00f,  1.3640350e+00f,  1.3390000e+00f,  1.3139350e+00f,
                1.4696500e+00f,  1.4446300e+00f,  1.4195800e+00f,  1.3945000e+00f,  1.3693900e+00f,  1.3442500e+00f,  1.3190800e+00f,
                1.4754250e+00f,  1.4503000e+00f,  1.4251450e+00f,  1.3999600e+00f,  1.3747450e+00f,  1.3495000e+00f,  1.3242250e+00f,
                1.4812000e+00f,  1.4559700e+00f,  1.4307100e+00f,  1.4054200e+00f,  1.3801000e+00f,  1.3547500e+00f,  1.3293700e+00f,
                1.4869750e+00f,  1.4616400e+00f,  1.4362750e+00f,  1.4108800e+00f,  1.3854550e+00f,  1.3600000e+00f,  1.3345150e+00f,
                1.4927500e+00f,  1.4673100e+00f,  1.4418400e+00f,  1.4163400e+00f,  1.3908100e+00f,  1.3652500e+00f,  1.3396600e+00f,
                1.4985250e+00f,  1.4729800e+00f,  1.4474050e+00f,  1.4218000e+00f,  1.3961650e+00f,  1.3705000e+00f,  1.3448050e+00f,
                9.3912000e-01f,  9.2202000e-01f,  9.0490000e-01f,  8.8776000e-01f,  8.7060000e-01f,  8.5342000e-01f,  8.3622000e-01f,
                4.3953000e-01f,  4.3094500e-01f,  4.2235000e-01f,  4.1374500e-01f,  4.0513000e-01f,  3.9650500e-01f,  3.8787000e-01f,
                5.6518000e-01f,  5.5663000e-01f,  5.4807000e-01f,  5.3950000e-01f,  5.3092000e-01f,  5.2233000e-01f,  5.1373000e-01f,
                1.0705450e+00f,  1.0533750e+00f,  1.0361850e+00f,  1.0189750e+00f,  1.0017450e+00f,  9.8449500e-01f,  9.6722500e-01f,
                1.5158500e+00f,  1.4899900e+00f,  1.4641000e+00f,  1.4381800e+00f,  1.4122300e+00f,  1.3862500e+00f,  1.3602400e+00f,
                1.5216250e+00f,  1.4956600e+00f,  1.4696650e+00f,  1.4436400e+00f,  1.4175850e+00f,  1.3915000e+00f,  1.3653850e+00f,
                1.5274000e+00f,  1.5013300e+00f,  1.4752300e+00f,  1.4491000e+00f,  1.4229400e+00f,  1.3967500e+00f,  1.3705300e+00f,
                1.5331750e+00f,  1.5070000e+00f,  1.4807950e+00f,  1.4545600e+00f,  1.4282950e+00f,  1.4020000e+00f,  1.3756750e+00f,
                1.5389500e+00f,  1.5126700e+00f,  1.4863600e+00f,  1.4600200e+00f,  1.4336500e+00f,  1.4072500e+00f,  1.3808200e+00f,
                1.5447250e+00f,  1.5183400e+00f,  1.4919250e+00f,  1.4654800e+00f,  1.4390050e+00f,  1.4125000e+00f,  1.3859650e+00f,
                1.5505000e+00f,  1.5240100e+00f,  1.4974900e+00f,  1.4709400e+00f,  1.4443600e+00f,  1.4177500e+00f,  1.3911100e+00f,
                1.5562750e+00f,  1.5296800e+00f,  1.5030550e+00f,  1.4764000e+00f,  1.4497150e+00f,  1.4230000e+00f,  1.3962550e+00f,
                1.5620500e+00f,  1.5353500e+00f,  1.5086200e+00f,  1.4818600e+00f,  1.4550700e+00f,  1.4282500e+00f,  1.4014000e+00f,
                9.7877500e-01f,  9.6090500e-01f,  9.4301500e-01f,  9.2510500e-01f,  9.0717500e-01f,  8.8922500e-01f,  8.7125500e-01f,
                4.5801000e-01f,  4.4904000e-01f,  4.4006000e-01f,  4.3107000e-01f,  4.2207000e-01f,  4.1306000e-01f,  4.0404000e-01f,
                3.8084200e-01f,  3.7380600e-01f,  3.6676200e-01f,  3.5971000e-01f,  3.5265000e-01f,  3.4558200e-01f,  3.3850600e-01f,
                7.1246000e-01f,  6.9833200e-01f,  6.8418800e-01f,  6.7002800e-01f,  6.5585200e-01f,  6.4166000e-01f,  6.2745200e-01f,
                9.9465800e-01f,  9.7338200e-01f,  9.5208200e-01f,  9.3075800e-01f,  9.0941000e-01f,  8.8803800e-01f,  8.6664200e-01f,
                9.9839600e-01f,  9.7703600e-01f,  9.5565200e-01f,  9.3424400e-01f,  9.1281200e-01f,  8.9135600e-01f,  8.6987600e-01f,
                1.0021340e+00f,  9.8069000e-01f,  9.5922200e-01f,  9.3773000e-01f,  9.1621400e-01f,  8.9467400e-01f,  8.7311000e-01f,
                1.0058720e+00f,  9.8434400e-01f,  9.6279200e-01f,  9.4121600e-01f,  9.1961600e-01f,  8.9799200e-01f,  8.7634400e-01f,
                1.0096100e+00f,  9.8799800e-01f,  9.6636200e-01f,  9.4470200e-01f,  9.2301800e-01f,  9.0131000e-01f,  8.7957800e-01f,
                1.0133480e+00f,  9.9165200e-01f,  9.6993200e-01f,  9.4818800e-01f,  9.2642000e-01f,  9.0462800e-01f,  8.8281200e-01f,
                1.0170860e+00f,  9.9530600e-01f,  9.7350200e-01f,  9.5167400e-01f,  9.2982200e-01f,  9.0794600e-01f,  8.8604600e-01f,
                1.0208240e+00f,  9.9896000e-01f,  9.7707200e-01f,  9.5516000e-01f,  9.3322400e-01f,  9.1126400e-01f,  8.8928000e-01f,
                1.0245620e+00f,  1.0026140e+00f,  9.8064200e-01f,  9.5864600e-01f,  9.3662600e-01f,  9.1458200e-01f,  8.9251400e-01f,
                6.3159600e-01f,  6.1690800e-01f,  6.0220400e-01f,  5.8748400e-01f,  5.7274800e-01f,  5.5799600e-01f,  5.4322800e-01f,
                2.9001000e-01f,  2.8263800e-01f,  2.7525800e-01f,  2.6787000e-01f,  2.6047400e-01f,  2.5307000e-01f,  2.4565800e-01f,
                2.3053800e-01f,  2.2511400e-01f,  2.1968400e-01f,  2.1424800e-01f,  2.0880600e-01f,  2.0335800e-01f,  1.9790400e-01f,
                4.2312900e-01f,  4.1223900e-01f,  4.0133700e-01f,  3.9042300e-01f,  3.7949700e-01f,  3.6855900e-01f,  3.5760900e-01f,
                5.7762600e-01f,  5.6122800e-01f,  5.4481200e-01f,  5.2837800e-01f,  5.1192600e-01f,  4.9545600e-01f,  4.7896800e-01f,
                5.7976800e-01f,  5.6330700e-01f,  5.4682800e-01f,  5.3033100e-01f,  5.1381600e-01f,  4.9728300e-01f,  4.8073200e-01f,
                5.8191000e-01f,  5.6538600e-01f,  5.4884400e-01f,  5.3228400e-01f,  5.1570600e-01f,  4.9911000e-01f,  4.8249600e-01f,
                5.8405200e-01f,  5.6746500e-01f,  5.5086000e-01f,  5.3423700e-01f,  5.1759600e-01f,  5.0093700e-01f,  4.8426000e-01f,
                5.8619400e-01f,  5.6954400e-01f,  5.5287600e-01f,  5.3619000e-01f,  5.1948600e-01f,  5.0276400e-01f,  4.8602400e-01f,
                5.8833600e-01f,  5.7162300e-01f,  5.5489200e-01f,  5.3814300e-01f,  5.2137600e-01f,  5.0459100e-01f,  4.8778800e-01f,
                5.9047800e-01f,  5.7370200e-01f,  5.5690800e-01f,  5.4009600e-01f,  5.2326600e-01f,  5.0641800e-01f,  4.8955200e-01f,
                5.9262000e-01f,  5.7578100e-01f,  5.5892400e-01f,  5.4204900e-01f,  5.2515600e-01f,  5.0824500e-01f,  4.9131600e-01f,
                5.9476200e-01f,  5.7786000e-01f,  5.6094000e-01f,  5.4400200e-01f,  5.2704600e-01f,  5.1007200e-01f,  4.9308000e-01f,
                3.5689500e-01f,  3.4558500e-01f,  3.3426300e-01f,  3.2292900e-01f,  3.1158300e-01f,  3.0022500e-01f,  2.8885500e-01f,
                1.5859200e-01f,  1.5291600e-01f,  1.4723400e-01f,  1.4154600e-01f,  1.3585200e-01f,  1.3015200e-01f,  1.2444600e-01f,
                1.1588500e-01f,  1.1217100e-01f,  1.0845300e-01f,  1.0473100e-01f,  1.0100500e-01f,  9.7275000e-02f,  9.3541000e-02f,
                2.0578600e-01f,  1.9833000e-01f,  1.9086600e-01f,  1.8339400e-01f,  1.7591400e-01f,  1.6842600e-01f,  1.6093000e-01f,
                2.6960500e-01f,  2.5837900e-01f,  2.4714100e-01f,  2.3589100e-01f,  2.2462900e-01f,  2.1335500e-01f,  2.0206900e-01f,
                2.7059200e-01f,  2.5932400e-01f,  2.4804400e-01f,  2.3675200e-01f,  2.2544800e-01f,  2.1413200e-01f,  2.0280400e-01f,
                2.7157900e-01f,  2.6026900e-01f,  2.4894700e-01f,  2.3761300e-01f,  2.2626700e-01f,  2.1490900e-01f,  2.0353900e-01f,
                2.7256600e-01f,  2.6121400e-01f,  2.4985000e-01f,  2.3847400e-01f,  2.2708600e-01f,  2.1568600e-01f,  2.0427400e-01f,
                2.7355300e-01f,  2.6215900e-01f,  2.5075300e-01f,  2.3933500e-01f,  2.2790500e-01f,  2.1646300e-01f,  2.0500900e-01f,
                2.7454000e-01f,  2.6310400e-01f,  2.5165600e-01f,  2.4019600e-01f,  2.2872400e-01f,  2.1724000e-01f,  2.0574400e-01f,
                2.7552700e-01f,  2.6404900e-01f,  2.5255900e-01f,  2.4105700e-01f,  2.2954300e-01f,  2.1801700e-01f,  2.0647900e-01f,
                2.7651400e-01f,  2.6499400e-01f,  2.5346200e-01f,  2.4191800e-01f,  2.3036200e-01f,  2.1879400e-01f,  2.0721400e-01f,
                2.7750100e-01f,  2.6593900e-01f,  2.5436500e-01f,  2.4277900e-01f,  2.3118100e-01f,  2.1957100e-01f,  2.0794900e-01f,
                1.5790600e-01f,  1.5017000e-01f,  1.4242600e-01f,  1.3467400e-01f,  1.2691400e-01f,  1.1914600e-01f,  1.1137000e-01f,
                6.5373000e-02f,  6.1491000e-02f,  5.7605000e-02f,  5.3715000e-02f,  4.9821000e-02f,  4.5923000e-02f,  4.2021000e-02f,
                3.8500000e-02f,  3.6594000e-02f,  3.4686000e-02f,  3.2776000e-02f,  3.0864000e-02f,  2.8950000e-02f,  2.7034000e-02f,
                6.3665000e-02f,  5.9839000e-02f,  5.6009000e-02f,  5.2175000e-02f,  4.8337000e-02f,  4.4495000e-02f,  4.0649000e-02f,
                7.5446000e-02f,  6.9686000e-02f,  6.3920000e-02f,  5.8148000e-02f,  5.2370000e-02f,  4.6586000e-02f,  4.0796000e-02f,
                7.5719000e-02f,  6.9938000e-02f,  6.4151000e-02f,  5.8358000e-02f,  5.2559000e-02f,  4.6754000e-02f,  4.0943000e-02f,
                7.5992000e-02f,  7.0190000e-02f,  6.4382000e-02f,  5.8568000e-02f,  5.2748000e-02f,  4.6922000e-02f,  4.1090000e-02f,
                7.6265000e-02f,  7.0442000e-02f,  6.4613000e-02f,  5.8778000e-02f,  5.2937000e-02f,  4.7090000e-02f,  4.1237000e-02f,
                7.6538000e-02f,  7.0694000e-02f,  6.4844000e-02f,  5.8988000e-02f,  5.3126000e-02f,  4.7258000e-02f,  4.1384000e-02f,
                7.6811000e-02f,  7.0946000e-02f,  6.5075000e-02f,  5.9198000e-02f,  5.3315000e-02f,  4.7426000e-02f,  4.1531000e-02f,
                7.7084000e-02f,  7.1198000e-02f,  6.5306000e-02f,  5.9408000e-02f,  5.3504000e-02f,  4.7594000e-02f,  4.1678000e-02f,
                7.7357000e-02f,  7.1450000e-02f,  6.5537000e-02f,  5.9618000e-02f,  5.3693000e-02f,  4.7762000e-02f,  4.1825000e-02f,
                7.7630000e-02f,  7.1702000e-02f,  6.5768000e-02f,  5.9828000e-02f,  5.3882000e-02f,  4.7930000e-02f,  4.1972000e-02f,
                3.7863000e-02f,  3.3897000e-02f,  2.9927000e-02f,  2.5953000e-02f,  2.1975000e-02f,  1.7993000e-02f,  1.4007000e-02f,
                1.1970000e-02f,  9.9800000e-03f,  7.9880000e-03f,  5.9940000e-03f,  3.9980000e-03f,  2.0000000e-03f,  0.0000000e+00f,
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
        }
    }
}
