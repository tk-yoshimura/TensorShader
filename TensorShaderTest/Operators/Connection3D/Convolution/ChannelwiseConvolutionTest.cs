using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class ChannelwiseConvolutionTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {
                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[kwidth * kheight * kdepth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map3D x = new(channels, inwidth, inheight, indepth, batch, xval);
                            Filter3D w = new(channels, 1, kwidth, kheight, kdepth, wval);

                            Map3D y = Reference(x, w, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(channels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(channels, 1, kwidth, kheight, kdepth), wval);

                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(channels, outwidth, outheight, outdepth, batch));

                            ChannelwiseConvolution ope = new(inwidth, inheight, indepth, channels, kwidth, kheight, kdepth, batch);

                            ope.Execute(x_tensor, w_tensor, y_tensor);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                            AssertError.Tolerance(y_expect, y_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {channels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

                            Console.WriteLine($"pass: {channels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
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
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {
                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[kwidth * kheight * kdepth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map3D x = new(channels, inwidth, inheight, indepth, batch, xval);
                            Filter3D w = new(channels, 1, kwidth, kheight, kdepth, wval);

                            Map3D y = Reference(x, w, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(channels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(channels, 1, kwidth, kheight, kdepth), wval);

                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(channels, outwidth, outheight, outdepth, batch));

                            ChannelwiseConvolution ope = new(inwidth, inheight, indepth, channels, kwidth, kheight, kdepth, batch);

                            ope.Execute(x_tensor, w_tensor, y_tensor);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

                            Console.WriteLine($"pass: {channels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
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
            int kwidth = 7, kheight = 5, kdepth = 3;
            int inwidth = 125, inheight = 196, indepth = 4;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[inwidth * inheight * indepth * channels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * kheight * kdepth * channels]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map3D x = new(channels, inwidth, inheight, indepth, batch, xval);
            Filter3D w = new(channels, 1, kwidth, kheight, kdepth, wval);

            Map3D y = Reference(x, w, kwidth, kheight, kdepth);

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(channels, inwidth, inheight, indepth, batch), xval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(channels, 1, kwidth, kheight, kdepth), wval);

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(channels, outwidth, outheight, outdepth, batch));

            ChannelwiseConvolution ope = new(inwidth, inheight, indepth, channels, kwidth, kheight, kdepth, batch);

            ope.Execute(x_tensor, w_tensor, y_tensor);

            float[] y_expect = y.ToArray();
            float[] y_actual = y_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

            Console.WriteLine($"pass: {channels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 64, inheight = 64, indepth = 64, channels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(channels, inwidth, inheight, indepth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(channels, 1, ksize, ksize, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(channels, outwidth, outheight, outdepth));

            ChannelwiseConvolution ope = new(inwidth, inheight, indepth, channels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/chwise_convolution_3d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 64, inheight = 64, indepth = 64, channels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(channels, inwidth, inheight, indepth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(channels, 1, ksize, ksize, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(channels, outwidth, outheight, outdepth));

            ChannelwiseConvolution ope = new(inwidth, inheight, indepth, channels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/chwise_convolution_3d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map3D Reference(Map3D x, Filter3D w, int kwidth, int kheight, int kdepth) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1, outd = ind - kdepth + 1;

            Map3D y = new(channels, outw, outh, outd, batch);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int ox, oy, oz = 0; oz < outd; oz++) {
                                for (oy = 0; oy < outh; oy++) {
                                    for (ox = 0; ox < outw; ox++) {
                                        for (int ch = 0; ch < channels; ch++) {
                                            y[ch, ox, oy, oz, th] += x[ch, kx + ox, ky + oy, kz + oz, th] * w[ch, 0, kx, ky, kz];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int channels = 2, kwidth = 3, kheight = 5, kdepth = 7, inwidth = 13, inheight = 12, indepth = 11, batch = 2;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[batch * inwidth * inheight * indepth * channels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * kdepth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map3D x = new(channels, inwidth, inheight, indepth, batch, xval);
            Filter3D w = new(channels, 1, kwidth, kheight, kdepth, wval);

            Map3D y = Reference(x, w, kwidth, kheight, kdepth);

            float[] y_expect = {
                6.9505100e+00f,  6.8574800e+00f,  6.9725600e+00f,  6.8793200e+00f,  6.9946100e+00f,  6.9011600e+00f,  7.0166600e+00f,  6.9230000e+00f,
                7.0387100e+00f,  6.9448400e+00f,  7.0607600e+00f,  6.9666800e+00f,  7.0828100e+00f,  6.9885200e+00f,  7.1048600e+00f,  7.0103600e+00f,
                7.1269100e+00f,  7.0322000e+00f,  7.1489600e+00f,  7.0540400e+00f,  7.1710100e+00f,  7.0758800e+00f,  7.2371600e+00f,  7.1414000e+00f,
                7.2592100e+00f,  7.1632400e+00f,  7.2812600e+00f,  7.1850800e+00f,  7.3033100e+00f,  7.2069200e+00f,  7.3253600e+00f,  7.2287600e+00f,
                7.3474100e+00f,  7.2506000e+00f,  7.3694600e+00f,  7.2724400e+00f,  7.3915100e+00f,  7.2942800e+00f,  7.4135600e+00f,  7.3161200e+00f,
                7.4356100e+00f,  7.3379600e+00f,  7.4576600e+00f,  7.3598000e+00f,  7.5238100e+00f,  7.4253200e+00f,  7.5458600e+00f,  7.4471600e+00f,
                7.5679100e+00f,  7.4690000e+00f,  7.5899600e+00f,  7.4908400e+00f,  7.6120100e+00f,  7.5126800e+00f,  7.6340600e+00f,  7.5345200e+00f,
                7.6561100e+00f,  7.5563600e+00f,  7.6781600e+00f,  7.5782000e+00f,  7.7002100e+00f,  7.6000400e+00f,  7.7222600e+00f,  7.6218800e+00f,
                7.7443100e+00f,  7.6437200e+00f,  7.8104600e+00f,  7.7092400e+00f,  7.8325100e+00f,  7.7310800e+00f,  7.8545600e+00f,  7.7529200e+00f,
                7.8766100e+00f,  7.7747600e+00f,  7.8986600e+00f,  7.7966000e+00f,  7.9207100e+00f,  7.8184400e+00f,  7.9427600e+00f,  7.8402800e+00f,
                7.9648100e+00f,  7.8621200e+00f,  7.9868600e+00f,  7.8839600e+00f,  8.0089100e+00f,  7.9058000e+00f,  8.0309600e+00f,  7.9276400e+00f,
                8.0971100e+00f,  7.9931600e+00f,  8.1191600e+00f,  8.0150000e+00f,  8.1412100e+00f,  8.0368400e+00f,  8.1632600e+00f,  8.0586800e+00f,
                8.1853100e+00f,  8.0805200e+00f,  8.2073600e+00f,  8.1023600e+00f,  8.2294100e+00f,  8.1242000e+00f,  8.2514600e+00f,  8.1460400e+00f,
                8.2735100e+00f,  8.1678800e+00f,  8.2955600e+00f,  8.1897200e+00f,  8.3176100e+00f,  8.2115600e+00f,  8.3837600e+00f,  8.2770800e+00f,
                8.4058100e+00f,  8.2989200e+00f,  8.4278600e+00f,  8.3207600e+00f,  8.4499100e+00f,  8.3426000e+00f,  8.4719600e+00f,  8.3644400e+00f,
                8.4940100e+00f,  8.3862800e+00f,  8.5160600e+00f,  8.4081200e+00f,  8.5381100e+00f,  8.4299600e+00f,  8.5601600e+00f,  8.4518000e+00f,
                8.5822100e+00f,  8.4736400e+00f,  8.6042600e+00f,  8.4954800e+00f,  8.6704100e+00f,  8.5610000e+00f,  8.6924600e+00f,  8.5828400e+00f,
                8.7145100e+00f,  8.6046800e+00f,  8.7365600e+00f,  8.6265200e+00f,  8.7586100e+00f,  8.6483600e+00f,  8.7806600e+00f,  8.6702000e+00f,
                8.8027100e+00f,  8.6920400e+00f,  8.8247600e+00f,  8.7138800e+00f,  8.8468100e+00f,  8.7357200e+00f,  8.8688600e+00f,  8.7575600e+00f,
                8.8909100e+00f,  8.7794000e+00f,  8.9570600e+00f,  8.8449200e+00f,  8.9791100e+00f,  8.8667600e+00f,  9.0011600e+00f,  8.8886000e+00f,
                9.0232100e+00f,  8.9104400e+00f,  9.0452600e+00f,  8.9322800e+00f,  9.0673100e+00f,  8.9541200e+00f,  9.0893600e+00f,  8.9759600e+00f,
                9.1114100e+00f,  8.9978000e+00f,  9.1334600e+00f,  9.0196400e+00f,  9.1555100e+00f,  9.0414800e+00f,  9.1775600e+00f,  9.0633200e+00f,
                1.0390310e+01f,  1.0264520e+01f,  1.0412360e+01f,  1.0286360e+01f,  1.0434410e+01f,  1.0308200e+01f,  1.0456460e+01f,  1.0330040e+01f,
                1.0478510e+01f,  1.0351880e+01f,  1.0500560e+01f,  1.0373720e+01f,  1.0522610e+01f,  1.0395560e+01f,  1.0544660e+01f,  1.0417400e+01f,
                1.0566710e+01f,  1.0439240e+01f,  1.0588760e+01f,  1.0461080e+01f,  1.0610810e+01f,  1.0482920e+01f,  1.0676960e+01f,  1.0548440e+01f,
                1.0699010e+01f,  1.0570280e+01f,  1.0721060e+01f,  1.0592120e+01f,  1.0743110e+01f,  1.0613960e+01f,  1.0765160e+01f,  1.0635800e+01f,
                1.0787210e+01f,  1.0657640e+01f,  1.0809260e+01f,  1.0679480e+01f,  1.0831310e+01f,  1.0701320e+01f,  1.0853360e+01f,  1.0723160e+01f,
                1.0875410e+01f,  1.0745000e+01f,  1.0897460e+01f,  1.0766840e+01f,  1.0963610e+01f,  1.0832360e+01f,  1.0985660e+01f,  1.0854200e+01f,
                1.1007710e+01f,  1.0876040e+01f,  1.1029760e+01f,  1.0897880e+01f,  1.1051810e+01f,  1.0919720e+01f,  1.1073860e+01f,  1.0941560e+01f,
                1.1095910e+01f,  1.0963400e+01f,  1.1117960e+01f,  1.0985240e+01f,  1.1140010e+01f,  1.1007080e+01f,  1.1162060e+01f,  1.1028920e+01f,
                1.1184110e+01f,  1.1050760e+01f,  1.1250260e+01f,  1.1116280e+01f,  1.1272310e+01f,  1.1138120e+01f,  1.1294360e+01f,  1.1159960e+01f,
                1.1316410e+01f,  1.1181800e+01f,  1.1338460e+01f,  1.1203640e+01f,  1.1360510e+01f,  1.1225480e+01f,  1.1382560e+01f,  1.1247320e+01f,
                1.1404610e+01f,  1.1269160e+01f,  1.1426660e+01f,  1.1291000e+01f,  1.1448710e+01f,  1.1312840e+01f,  1.1470760e+01f,  1.1334680e+01f,
                1.1536910e+01f,  1.1400200e+01f,  1.1558960e+01f,  1.1422040e+01f,  1.1581010e+01f,  1.1443880e+01f,  1.1603060e+01f,  1.1465720e+01f,
                1.1625110e+01f,  1.1487560e+01f,  1.1647160e+01f,  1.1509400e+01f,  1.1669210e+01f,  1.1531240e+01f,  1.1691260e+01f,  1.1553080e+01f,
                1.1713310e+01f,  1.1574920e+01f,  1.1735360e+01f,  1.1596760e+01f,  1.1757410e+01f,  1.1618600e+01f,  1.1823560e+01f,  1.1684120e+01f,
                1.1845610e+01f,  1.1705960e+01f,  1.1867660e+01f,  1.1727800e+01f,  1.1889710e+01f,  1.1749640e+01f,  1.1911760e+01f,  1.1771480e+01f,
                1.1933810e+01f,  1.1793320e+01f,  1.1955860e+01f,  1.1815160e+01f,  1.1977910e+01f,  1.1837000e+01f,  1.1999960e+01f,  1.1858840e+01f,
                1.2022010e+01f,  1.1880680e+01f,  1.2044060e+01f,  1.1902520e+01f,  1.2110210e+01f,  1.1968040e+01f,  1.2132260e+01f,  1.1989880e+01f,
                1.2154310e+01f,  1.2011720e+01f,  1.2176360e+01f,  1.2033560e+01f,  1.2198410e+01f,  1.2055400e+01f,  1.2220460e+01f,  1.2077240e+01f,
                1.2242510e+01f,  1.2099080e+01f,  1.2264560e+01f,  1.2120920e+01f,  1.2286610e+01f,  1.2142760e+01f,  1.2308660e+01f,  1.2164600e+01f,
                1.2330710e+01f,  1.2186440e+01f,  1.2396860e+01f,  1.2251960e+01f,  1.2418910e+01f,  1.2273800e+01f,  1.2440960e+01f,  1.2295640e+01f,
                1.2463010e+01f,  1.2317480e+01f,  1.2485060e+01f,  1.2339320e+01f,  1.2507110e+01f,  1.2361160e+01f,  1.2529160e+01f,  1.2383000e+01f,
                1.2551210e+01f,  1.2404840e+01f,  1.2573260e+01f,  1.2426680e+01f,  1.2595310e+01f,  1.2448520e+01f,  1.2617360e+01f,  1.2470360e+01f,
                1.3830110e+01f,  1.3671560e+01f,  1.3852160e+01f,  1.3693400e+01f,  1.3874210e+01f,  1.3715240e+01f,  1.3896260e+01f,  1.3737080e+01f,
                1.3918310e+01f,  1.3758920e+01f,  1.3940360e+01f,  1.3780760e+01f,  1.3962410e+01f,  1.3802600e+01f,  1.3984460e+01f,  1.3824440e+01f,
                1.4006510e+01f,  1.3846280e+01f,  1.4028560e+01f,  1.3868120e+01f,  1.4050610e+01f,  1.3889960e+01f,  1.4116760e+01f,  1.3955480e+01f,
                1.4138810e+01f,  1.3977320e+01f,  1.4160860e+01f,  1.3999160e+01f,  1.4182910e+01f,  1.4021000e+01f,  1.4204960e+01f,  1.4042840e+01f,
                1.4227010e+01f,  1.4064680e+01f,  1.4249060e+01f,  1.4086520e+01f,  1.4271110e+01f,  1.4108360e+01f,  1.4293160e+01f,  1.4130200e+01f,
                1.4315210e+01f,  1.4152040e+01f,  1.4337260e+01f,  1.4173880e+01f,  1.4403410e+01f,  1.4239400e+01f,  1.4425460e+01f,  1.4261240e+01f,
                1.4447510e+01f,  1.4283080e+01f,  1.4469560e+01f,  1.4304920e+01f,  1.4491610e+01f,  1.4326760e+01f,  1.4513660e+01f,  1.4348600e+01f,
                1.4535710e+01f,  1.4370440e+01f,  1.4557760e+01f,  1.4392280e+01f,  1.4579810e+01f,  1.4414120e+01f,  1.4601860e+01f,  1.4435960e+01f,
                1.4623910e+01f,  1.4457800e+01f,  1.4690060e+01f,  1.4523320e+01f,  1.4712110e+01f,  1.4545160e+01f,  1.4734160e+01f,  1.4567000e+01f,
                1.4756210e+01f,  1.4588840e+01f,  1.4778260e+01f,  1.4610680e+01f,  1.4800310e+01f,  1.4632520e+01f,  1.4822360e+01f,  1.4654360e+01f,
                1.4844410e+01f,  1.4676200e+01f,  1.4866460e+01f,  1.4698040e+01f,  1.4888510e+01f,  1.4719880e+01f,  1.4910560e+01f,  1.4741720e+01f,
                1.4976710e+01f,  1.4807240e+01f,  1.4998760e+01f,  1.4829080e+01f,  1.5020810e+01f,  1.4850920e+01f,  1.5042860e+01f,  1.4872760e+01f,
                1.5064910e+01f,  1.4894600e+01f,  1.5086960e+01f,  1.4916440e+01f,  1.5109010e+01f,  1.4938280e+01f,  1.5131060e+01f,  1.4960120e+01f,
                1.5153110e+01f,  1.4981960e+01f,  1.5175160e+01f,  1.5003800e+01f,  1.5197210e+01f,  1.5025640e+01f,  1.5263360e+01f,  1.5091160e+01f,
                1.5285410e+01f,  1.5113000e+01f,  1.5307460e+01f,  1.5134840e+01f,  1.5329510e+01f,  1.5156680e+01f,  1.5351560e+01f,  1.5178520e+01f,
                1.5373610e+01f,  1.5200360e+01f,  1.5395660e+01f,  1.5222200e+01f,  1.5417710e+01f,  1.5244040e+01f,  1.5439760e+01f,  1.5265880e+01f,
                1.5461810e+01f,  1.5287720e+01f,  1.5483860e+01f,  1.5309560e+01f,  1.5550010e+01f,  1.5375080e+01f,  1.5572060e+01f,  1.5396920e+01f,
                1.5594110e+01f,  1.5418760e+01f,  1.5616160e+01f,  1.5440600e+01f,  1.5638210e+01f,  1.5462440e+01f,  1.5660260e+01f,  1.5484280e+01f,
                1.5682310e+01f,  1.5506120e+01f,  1.5704360e+01f,  1.5527960e+01f,  1.5726410e+01f,  1.5549800e+01f,  1.5748460e+01f,  1.5571640e+01f,
                1.5770510e+01f,  1.5593480e+01f,  1.5836660e+01f,  1.5659000e+01f,  1.5858710e+01f,  1.5680840e+01f,  1.5880760e+01f,  1.5702680e+01f,
                1.5902810e+01f,  1.5724520e+01f,  1.5924860e+01f,  1.5746360e+01f,  1.5946910e+01f,  1.5768200e+01f,  1.5968960e+01f,  1.5790040e+01f,
                1.5991010e+01f,  1.5811880e+01f,  1.6013060e+01f,  1.5833720e+01f,  1.6035110e+01f,  1.5855560e+01f,  1.6057160e+01f,  1.5877400e+01f,
                1.7269910e+01f,  1.7078600e+01f,  1.7291960e+01f,  1.7100440e+01f,  1.7314010e+01f,  1.7122280e+01f,  1.7336060e+01f,  1.7144120e+01f,
                1.7358110e+01f,  1.7165960e+01f,  1.7380160e+01f,  1.7187800e+01f,  1.7402210e+01f,  1.7209640e+01f,  1.7424260e+01f,  1.7231480e+01f,
                1.7446310e+01f,  1.7253320e+01f,  1.7468360e+01f,  1.7275160e+01f,  1.7490410e+01f,  1.7297000e+01f,  1.7556560e+01f,  1.7362520e+01f,
                1.7578610e+01f,  1.7384360e+01f,  1.7600660e+01f,  1.7406200e+01f,  1.7622710e+01f,  1.7428040e+01f,  1.7644760e+01f,  1.7449880e+01f,
                1.7666810e+01f,  1.7471720e+01f,  1.7688860e+01f,  1.7493560e+01f,  1.7710910e+01f,  1.7515400e+01f,  1.7732960e+01f,  1.7537240e+01f,
                1.7755010e+01f,  1.7559080e+01f,  1.7777060e+01f,  1.7580920e+01f,  1.7843210e+01f,  1.7646440e+01f,  1.7865260e+01f,  1.7668280e+01f,
                1.7887310e+01f,  1.7690120e+01f,  1.7909360e+01f,  1.7711960e+01f,  1.7931410e+01f,  1.7733800e+01f,  1.7953460e+01f,  1.7755640e+01f,
                1.7975510e+01f,  1.7777480e+01f,  1.7997560e+01f,  1.7799320e+01f,  1.8019610e+01f,  1.7821160e+01f,  1.8041660e+01f,  1.7843000e+01f,
                1.8063710e+01f,  1.7864840e+01f,  1.8129860e+01f,  1.7930360e+01f,  1.8151910e+01f,  1.7952200e+01f,  1.8173960e+01f,  1.7974040e+01f,
                1.8196010e+01f,  1.7995880e+01f,  1.8218060e+01f,  1.8017720e+01f,  1.8240110e+01f,  1.8039560e+01f,  1.8262160e+01f,  1.8061400e+01f,
                1.8284210e+01f,  1.8083240e+01f,  1.8306260e+01f,  1.8105080e+01f,  1.8328310e+01f,  1.8126920e+01f,  1.8350360e+01f,  1.8148760e+01f,
                1.8416510e+01f,  1.8214280e+01f,  1.8438560e+01f,  1.8236120e+01f,  1.8460610e+01f,  1.8257960e+01f,  1.8482660e+01f,  1.8279800e+01f,
                1.8504710e+01f,  1.8301640e+01f,  1.8526760e+01f,  1.8323480e+01f,  1.8548810e+01f,  1.8345320e+01f,  1.8570860e+01f,  1.8367160e+01f,
                1.8592910e+01f,  1.8389000e+01f,  1.8614960e+01f,  1.8410840e+01f,  1.8637010e+01f,  1.8432680e+01f,  1.8703160e+01f,  1.8498200e+01f,
                1.8725210e+01f,  1.8520040e+01f,  1.8747260e+01f,  1.8541880e+01f,  1.8769310e+01f,  1.8563720e+01f,  1.8791360e+01f,  1.8585560e+01f,
                1.8813410e+01f,  1.8607400e+01f,  1.8835460e+01f,  1.8629240e+01f,  1.8857510e+01f,  1.8651080e+01f,  1.8879560e+01f,  1.8672920e+01f,
                1.8901610e+01f,  1.8694760e+01f,  1.8923660e+01f,  1.8716600e+01f,  1.8989810e+01f,  1.8782120e+01f,  1.9011860e+01f,  1.8803960e+01f,
                1.9033910e+01f,  1.8825800e+01f,  1.9055960e+01f,  1.8847640e+01f,  1.9078010e+01f,  1.8869480e+01f,  1.9100060e+01f,  1.8891320e+01f,
                1.9122110e+01f,  1.8913160e+01f,  1.9144160e+01f,  1.8935000e+01f,  1.9166210e+01f,  1.8956840e+01f,  1.9188260e+01f,  1.8978680e+01f,
                1.9210310e+01f,  1.9000520e+01f,  1.9276460e+01f,  1.9066040e+01f,  1.9298510e+01f,  1.9087880e+01f,  1.9320560e+01f,  1.9109720e+01f,
                1.9342610e+01f,  1.9131560e+01f,  1.9364660e+01f,  1.9153400e+01f,  1.9386710e+01f,  1.9175240e+01f,  1.9408760e+01f,  1.9197080e+01f,
                1.9430810e+01f,  1.9218920e+01f,  1.9452860e+01f,  1.9240760e+01f,  1.9474910e+01f,  1.9262600e+01f,  1.9496960e+01f,  1.9284440e+01f,
                2.0709710e+01f,  2.0485640e+01f,  2.0731760e+01f,  2.0507480e+01f,  2.0753810e+01f,  2.0529320e+01f,  2.0775860e+01f,  2.0551160e+01f,
                2.0797910e+01f,  2.0573000e+01f,  2.0819960e+01f,  2.0594840e+01f,  2.0842010e+01f,  2.0616680e+01f,  2.0864060e+01f,  2.0638520e+01f,
                2.0886110e+01f,  2.0660360e+01f,  2.0908160e+01f,  2.0682200e+01f,  2.0930210e+01f,  2.0704040e+01f,  2.0996360e+01f,  2.0769560e+01f,
                2.1018410e+01f,  2.0791400e+01f,  2.1040460e+01f,  2.0813240e+01f,  2.1062510e+01f,  2.0835080e+01f,  2.1084560e+01f,  2.0856920e+01f,
                2.1106610e+01f,  2.0878760e+01f,  2.1128660e+01f,  2.0900600e+01f,  2.1150710e+01f,  2.0922440e+01f,  2.1172760e+01f,  2.0944280e+01f,
                2.1194810e+01f,  2.0966120e+01f,  2.1216860e+01f,  2.0987960e+01f,  2.1283010e+01f,  2.1053480e+01f,  2.1305060e+01f,  2.1075320e+01f,
                2.1327110e+01f,  2.1097160e+01f,  2.1349160e+01f,  2.1119000e+01f,  2.1371210e+01f,  2.1140840e+01f,  2.1393260e+01f,  2.1162680e+01f,
                2.1415310e+01f,  2.1184520e+01f,  2.1437360e+01f,  2.1206360e+01f,  2.1459410e+01f,  2.1228200e+01f,  2.1481460e+01f,  2.1250040e+01f,
                2.1503510e+01f,  2.1271880e+01f,  2.1569660e+01f,  2.1337400e+01f,  2.1591710e+01f,  2.1359240e+01f,  2.1613760e+01f,  2.1381080e+01f,
                2.1635810e+01f,  2.1402920e+01f,  2.1657860e+01f,  2.1424760e+01f,  2.1679910e+01f,  2.1446600e+01f,  2.1701960e+01f,  2.1468440e+01f,
                2.1724010e+01f,  2.1490280e+01f,  2.1746060e+01f,  2.1512120e+01f,  2.1768110e+01f,  2.1533960e+01f,  2.1790160e+01f,  2.1555800e+01f,
                2.1856310e+01f,  2.1621320e+01f,  2.1878360e+01f,  2.1643160e+01f,  2.1900410e+01f,  2.1665000e+01f,  2.1922460e+01f,  2.1686840e+01f,
                2.1944510e+01f,  2.1708680e+01f,  2.1966560e+01f,  2.1730520e+01f,  2.1988610e+01f,  2.1752360e+01f,  2.2010660e+01f,  2.1774200e+01f,
                2.2032710e+01f,  2.1796040e+01f,  2.2054760e+01f,  2.1817880e+01f,  2.2076810e+01f,  2.1839720e+01f,  2.2142960e+01f,  2.1905240e+01f,
                2.2165010e+01f,  2.1927080e+01f,  2.2187060e+01f,  2.1948920e+01f,  2.2209110e+01f,  2.1970760e+01f,  2.2231160e+01f,  2.1992600e+01f,
                2.2253210e+01f,  2.2014440e+01f,  2.2275260e+01f,  2.2036280e+01f,  2.2297310e+01f,  2.2058120e+01f,  2.2319360e+01f,  2.2079960e+01f,
                2.2341410e+01f,  2.2101800e+01f,  2.2363460e+01f,  2.2123640e+01f,  2.2429610e+01f,  2.2189160e+01f,  2.2451660e+01f,  2.2211000e+01f,
                2.2473710e+01f,  2.2232840e+01f,  2.2495760e+01f,  2.2254680e+01f,  2.2517810e+01f,  2.2276520e+01f,  2.2539860e+01f,  2.2298360e+01f,
                2.2561910e+01f,  2.2320200e+01f,  2.2583960e+01f,  2.2342040e+01f,  2.2606010e+01f,  2.2363880e+01f,  2.2628060e+01f,  2.2385720e+01f,
                2.2650110e+01f,  2.2407560e+01f,  2.2716260e+01f,  2.2473080e+01f,  2.2738310e+01f,  2.2494920e+01f,  2.2760360e+01f,  2.2516760e+01f,
                2.2782410e+01f,  2.2538600e+01f,  2.2804460e+01f,  2.2560440e+01f,  2.2826510e+01f,  2.2582280e+01f,  2.2848560e+01f,  2.2604120e+01f,
                2.2870610e+01f,  2.2625960e+01f,  2.2892660e+01f,  2.2647800e+01f,  2.2914710e+01f,  2.2669640e+01f,  2.2936760e+01f,  2.2691480e+01f,
                4.4788310e+01f,  4.4334920e+01f,  4.4810360e+01f,  4.4356760e+01f,  4.4832410e+01f,  4.4378600e+01f,  4.4854460e+01f,  4.4400440e+01f,
                4.4876510e+01f,  4.4422280e+01f,  4.4898560e+01f,  4.4444120e+01f,  4.4920610e+01f,  4.4465960e+01f,  4.4942660e+01f,  4.4487800e+01f,
                4.4964710e+01f,  4.4509640e+01f,  4.4986760e+01f,  4.4531480e+01f,  4.5008810e+01f,  4.4553320e+01f,  4.5074960e+01f,  4.4618840e+01f,
                4.5097010e+01f,  4.4640680e+01f,  4.5119060e+01f,  4.4662520e+01f,  4.5141110e+01f,  4.4684360e+01f,  4.5163160e+01f,  4.4706200e+01f,
                4.5185210e+01f,  4.4728040e+01f,  4.5207260e+01f,  4.4749880e+01f,  4.5229310e+01f,  4.4771720e+01f,  4.5251360e+01f,  4.4793560e+01f,
                4.5273410e+01f,  4.4815400e+01f,  4.5295460e+01f,  4.4837240e+01f,  4.5361610e+01f,  4.4902760e+01f,  4.5383660e+01f,  4.4924600e+01f,
                4.5405710e+01f,  4.4946440e+01f,  4.5427760e+01f,  4.4968280e+01f,  4.5449810e+01f,  4.4990120e+01f,  4.5471860e+01f,  4.5011960e+01f,
                4.5493910e+01f,  4.5033800e+01f,  4.5515960e+01f,  4.5055640e+01f,  4.5538010e+01f,  4.5077480e+01f,  4.5560060e+01f,  4.5099320e+01f,
                4.5582110e+01f,  4.5121160e+01f,  4.5648260e+01f,  4.5186680e+01f,  4.5670310e+01f,  4.5208520e+01f,  4.5692360e+01f,  4.5230360e+01f,
                4.5714410e+01f,  4.5252200e+01f,  4.5736460e+01f,  4.5274040e+01f,  4.5758510e+01f,  4.5295880e+01f,  4.5780560e+01f,  4.5317720e+01f,
                4.5802610e+01f,  4.5339560e+01f,  4.5824660e+01f,  4.5361400e+01f,  4.5846710e+01f,  4.5383240e+01f,  4.5868760e+01f,  4.5405080e+01f,
                4.5934910e+01f,  4.5470600e+01f,  4.5956960e+01f,  4.5492440e+01f,  4.5979010e+01f,  4.5514280e+01f,  4.6001060e+01f,  4.5536120e+01f,
                4.6023110e+01f,  4.5557960e+01f,  4.6045160e+01f,  4.5579800e+01f,  4.6067210e+01f,  4.5601640e+01f,  4.6089260e+01f,  4.5623480e+01f,
                4.6111310e+01f,  4.5645320e+01f,  4.6133360e+01f,  4.5667160e+01f,  4.6155410e+01f,  4.5689000e+01f,  4.6221560e+01f,  4.5754520e+01f,
                4.6243610e+01f,  4.5776360e+01f,  4.6265660e+01f,  4.5798200e+01f,  4.6287710e+01f,  4.5820040e+01f,  4.6309760e+01f,  4.5841880e+01f,
                4.6331810e+01f,  4.5863720e+01f,  4.6353860e+01f,  4.5885560e+01f,  4.6375910e+01f,  4.5907400e+01f,  4.6397960e+01f,  4.5929240e+01f,
                4.6420010e+01f,  4.5951080e+01f,  4.6442060e+01f,  4.5972920e+01f,  4.6508210e+01f,  4.6038440e+01f,  4.6530260e+01f,  4.6060280e+01f,
                4.6552310e+01f,  4.6082120e+01f,  4.6574360e+01f,  4.6103960e+01f,  4.6596410e+01f,  4.6125800e+01f,  4.6618460e+01f,  4.6147640e+01f,
                4.6640510e+01f,  4.6169480e+01f,  4.6662560e+01f,  4.6191320e+01f,  4.6684610e+01f,  4.6213160e+01f,  4.6706660e+01f,  4.6235000e+01f,
                4.6728710e+01f,  4.6256840e+01f,  4.6794860e+01f,  4.6322360e+01f,  4.6816910e+01f,  4.6344200e+01f,  4.6838960e+01f,  4.6366040e+01f,
                4.6861010e+01f,  4.6387880e+01f,  4.6883060e+01f,  4.6409720e+01f,  4.6905110e+01f,  4.6431560e+01f,  4.6927160e+01f,  4.6453400e+01f,
                4.6949210e+01f,  4.6475240e+01f,  4.6971260e+01f,  4.6497080e+01f,  4.6993310e+01f,  4.6518920e+01f,  4.7015360e+01f,  4.6540760e+01f,
                4.8228110e+01f,  4.7741960e+01f,  4.8250160e+01f,  4.7763800e+01f,  4.8272210e+01f,  4.7785640e+01f,  4.8294260e+01f,  4.7807480e+01f,
                4.8316310e+01f,  4.7829320e+01f,  4.8338360e+01f,  4.7851160e+01f,  4.8360410e+01f,  4.7873000e+01f,  4.8382460e+01f,  4.7894840e+01f,
                4.8404510e+01f,  4.7916680e+01f,  4.8426560e+01f,  4.7938520e+01f,  4.8448610e+01f,  4.7960360e+01f,  4.8514760e+01f,  4.8025880e+01f,
                4.8536810e+01f,  4.8047720e+01f,  4.8558860e+01f,  4.8069560e+01f,  4.8580910e+01f,  4.8091400e+01f,  4.8602960e+01f,  4.8113240e+01f,
                4.8625010e+01f,  4.8135080e+01f,  4.8647060e+01f,  4.8156920e+01f,  4.8669110e+01f,  4.8178760e+01f,  4.8691160e+01f,  4.8200600e+01f,
                4.8713210e+01f,  4.8222440e+01f,  4.8735260e+01f,  4.8244280e+01f,  4.8801410e+01f,  4.8309800e+01f,  4.8823460e+01f,  4.8331640e+01f,
                4.8845510e+01f,  4.8353480e+01f,  4.8867560e+01f,  4.8375320e+01f,  4.8889610e+01f,  4.8397160e+01f,  4.8911660e+01f,  4.8419000e+01f,
                4.8933710e+01f,  4.8440840e+01f,  4.8955760e+01f,  4.8462680e+01f,  4.8977810e+01f,  4.8484520e+01f,  4.8999860e+01f,  4.8506360e+01f,
                4.9021910e+01f,  4.8528200e+01f,  4.9088060e+01f,  4.8593720e+01f,  4.9110110e+01f,  4.8615560e+01f,  4.9132160e+01f,  4.8637400e+01f,
                4.9154210e+01f,  4.8659240e+01f,  4.9176260e+01f,  4.8681080e+01f,  4.9198310e+01f,  4.8702920e+01f,  4.9220360e+01f,  4.8724760e+01f,
                4.9242410e+01f,  4.8746600e+01f,  4.9264460e+01f,  4.8768440e+01f,  4.9286510e+01f,  4.8790280e+01f,  4.9308560e+01f,  4.8812120e+01f,
                4.9374710e+01f,  4.8877640e+01f,  4.9396760e+01f,  4.8899480e+01f,  4.9418810e+01f,  4.8921320e+01f,  4.9440860e+01f,  4.8943160e+01f,
                4.9462910e+01f,  4.8965000e+01f,  4.9484960e+01f,  4.8986840e+01f,  4.9507010e+01f,  4.9008680e+01f,  4.9529060e+01f,  4.9030520e+01f,
                4.9551110e+01f,  4.9052360e+01f,  4.9573160e+01f,  4.9074200e+01f,  4.9595210e+01f,  4.9096040e+01f,  4.9661360e+01f,  4.9161560e+01f,
                4.9683410e+01f,  4.9183400e+01f,  4.9705460e+01f,  4.9205240e+01f,  4.9727510e+01f,  4.9227080e+01f,  4.9749560e+01f,  4.9248920e+01f,
                4.9771610e+01f,  4.9270760e+01f,  4.9793660e+01f,  4.9292600e+01f,  4.9815710e+01f,  4.9314440e+01f,  4.9837760e+01f,  4.9336280e+01f,
                4.9859810e+01f,  4.9358120e+01f,  4.9881860e+01f,  4.9379960e+01f,  4.9948010e+01f,  4.9445480e+01f,  4.9970060e+01f,  4.9467320e+01f,
                4.9992110e+01f,  4.9489160e+01f,  5.0014160e+01f,  4.9511000e+01f,  5.0036210e+01f,  4.9532840e+01f,  5.0058260e+01f,  4.9554680e+01f,
                5.0080310e+01f,  4.9576520e+01f,  5.0102360e+01f,  4.9598360e+01f,  5.0124410e+01f,  4.9620200e+01f,  5.0146460e+01f,  4.9642040e+01f,
                5.0168510e+01f,  4.9663880e+01f,  5.0234660e+01f,  4.9729400e+01f,  5.0256710e+01f,  4.9751240e+01f,  5.0278760e+01f,  4.9773080e+01f,
                5.0300810e+01f,  4.9794920e+01f,  5.0322860e+01f,  4.9816760e+01f,  5.0344910e+01f,  4.9838600e+01f,  5.0366960e+01f,  4.9860440e+01f,
                5.0389010e+01f,  4.9882280e+01f,  5.0411060e+01f,  4.9904120e+01f,  5.0433110e+01f,  4.9925960e+01f,  5.0455160e+01f,  4.9947800e+01f,
                5.1667910e+01f,  5.1149000e+01f,  5.1689960e+01f,  5.1170840e+01f,  5.1712010e+01f,  5.1192680e+01f,  5.1734060e+01f,  5.1214520e+01f,
                5.1756110e+01f,  5.1236360e+01f,  5.1778160e+01f,  5.1258200e+01f,  5.1800210e+01f,  5.1280040e+01f,  5.1822260e+01f,  5.1301880e+01f,
                5.1844310e+01f,  5.1323720e+01f,  5.1866360e+01f,  5.1345560e+01f,  5.1888410e+01f,  5.1367400e+01f,  5.1954560e+01f,  5.1432920e+01f,
                5.1976610e+01f,  5.1454760e+01f,  5.1998660e+01f,  5.1476600e+01f,  5.2020710e+01f,  5.1498440e+01f,  5.2042760e+01f,  5.1520280e+01f,
                5.2064810e+01f,  5.1542120e+01f,  5.2086860e+01f,  5.1563960e+01f,  5.2108910e+01f,  5.1585800e+01f,  5.2130960e+01f,  5.1607640e+01f,
                5.2153010e+01f,  5.1629480e+01f,  5.2175060e+01f,  5.1651320e+01f,  5.2241210e+01f,  5.1716840e+01f,  5.2263260e+01f,  5.1738680e+01f,
                5.2285310e+01f,  5.1760520e+01f,  5.2307360e+01f,  5.1782360e+01f,  5.2329410e+01f,  5.1804200e+01f,  5.2351460e+01f,  5.1826040e+01f,
                5.2373510e+01f,  5.1847880e+01f,  5.2395560e+01f,  5.1869720e+01f,  5.2417610e+01f,  5.1891560e+01f,  5.2439660e+01f,  5.1913400e+01f,
                5.2461710e+01f,  5.1935240e+01f,  5.2527860e+01f,  5.2000760e+01f,  5.2549910e+01f,  5.2022600e+01f,  5.2571960e+01f,  5.2044440e+01f,
                5.2594010e+01f,  5.2066280e+01f,  5.2616060e+01f,  5.2088120e+01f,  5.2638110e+01f,  5.2109960e+01f,  5.2660160e+01f,  5.2131800e+01f,
                5.2682210e+01f,  5.2153640e+01f,  5.2704260e+01f,  5.2175480e+01f,  5.2726310e+01f,  5.2197320e+01f,  5.2748360e+01f,  5.2219160e+01f,
                5.2814510e+01f,  5.2284680e+01f,  5.2836560e+01f,  5.2306520e+01f,  5.2858610e+01f,  5.2328360e+01f,  5.2880660e+01f,  5.2350200e+01f,
                5.2902710e+01f,  5.2372040e+01f,  5.2924760e+01f,  5.2393880e+01f,  5.2946810e+01f,  5.2415720e+01f,  5.2968860e+01f,  5.2437560e+01f,
                5.2990910e+01f,  5.2459400e+01f,  5.3012960e+01f,  5.2481240e+01f,  5.3035010e+01f,  5.2503080e+01f,  5.3101160e+01f,  5.2568600e+01f,
                5.3123210e+01f,  5.2590440e+01f,  5.3145260e+01f,  5.2612280e+01f,  5.3167310e+01f,  5.2634120e+01f,  5.3189360e+01f,  5.2655960e+01f,
                5.3211410e+01f,  5.2677800e+01f,  5.3233460e+01f,  5.2699640e+01f,  5.3255510e+01f,  5.2721480e+01f,  5.3277560e+01f,  5.2743320e+01f,
                5.3299610e+01f,  5.2765160e+01f,  5.3321660e+01f,  5.2787000e+01f,  5.3387810e+01f,  5.2852520e+01f,  5.3409860e+01f,  5.2874360e+01f,
                5.3431910e+01f,  5.2896200e+01f,  5.3453960e+01f,  5.2918040e+01f,  5.3476010e+01f,  5.2939880e+01f,  5.3498060e+01f,  5.2961720e+01f,
                5.3520110e+01f,  5.2983560e+01f,  5.3542160e+01f,  5.3005400e+01f,  5.3564210e+01f,  5.3027240e+01f,  5.3586260e+01f,  5.3049080e+01f,
                5.3608310e+01f,  5.3070920e+01f,  5.3674460e+01f,  5.3136440e+01f,  5.3696510e+01f,  5.3158280e+01f,  5.3718560e+01f,  5.3180120e+01f,
                5.3740610e+01f,  5.3201960e+01f,  5.3762660e+01f,  5.3223800e+01f,  5.3784710e+01f,  5.3245640e+01f,  5.3806760e+01f,  5.3267480e+01f,
                5.3828810e+01f,  5.3289320e+01f,  5.3850860e+01f,  5.3311160e+01f,  5.3872910e+01f,  5.3333000e+01f,  5.3894960e+01f,  5.3354840e+01f,
                5.5107710e+01f,  5.4556040e+01f,  5.5129760e+01f,  5.4577880e+01f,  5.5151810e+01f,  5.4599720e+01f,  5.5173860e+01f,  5.4621560e+01f,
                5.5195910e+01f,  5.4643400e+01f,  5.5217960e+01f,  5.4665240e+01f,  5.5240010e+01f,  5.4687080e+01f,  5.5262060e+01f,  5.4708920e+01f,
                5.5284110e+01f,  5.4730760e+01f,  5.5306160e+01f,  5.4752600e+01f,  5.5328210e+01f,  5.4774440e+01f,  5.5394360e+01f,  5.4839960e+01f,
                5.5416410e+01f,  5.4861800e+01f,  5.5438460e+01f,  5.4883640e+01f,  5.5460510e+01f,  5.4905480e+01f,  5.5482560e+01f,  5.4927320e+01f,
                5.5504610e+01f,  5.4949160e+01f,  5.5526660e+01f,  5.4971000e+01f,  5.5548710e+01f,  5.4992840e+01f,  5.5570760e+01f,  5.5014680e+01f,
                5.5592810e+01f,  5.5036520e+01f,  5.5614860e+01f,  5.5058360e+01f,  5.5681010e+01f,  5.5123880e+01f,  5.5703060e+01f,  5.5145720e+01f,
                5.5725110e+01f,  5.5167560e+01f,  5.5747160e+01f,  5.5189400e+01f,  5.5769210e+01f,  5.5211240e+01f,  5.5791260e+01f,  5.5233080e+01f,
                5.5813310e+01f,  5.5254920e+01f,  5.5835360e+01f,  5.5276760e+01f,  5.5857410e+01f,  5.5298600e+01f,  5.5879460e+01f,  5.5320440e+01f,
                5.5901510e+01f,  5.5342280e+01f,  5.5967660e+01f,  5.5407800e+01f,  5.5989710e+01f,  5.5429640e+01f,  5.6011760e+01f,  5.5451480e+01f,
                5.6033810e+01f,  5.5473320e+01f,  5.6055860e+01f,  5.5495160e+01f,  5.6077910e+01f,  5.5517000e+01f,  5.6099960e+01f,  5.5538840e+01f,
                5.6122010e+01f,  5.5560680e+01f,  5.6144060e+01f,  5.5582520e+01f,  5.6166110e+01f,  5.5604360e+01f,  5.6188160e+01f,  5.5626200e+01f,
                5.6254310e+01f,  5.5691720e+01f,  5.6276360e+01f,  5.5713560e+01f,  5.6298410e+01f,  5.5735400e+01f,  5.6320460e+01f,  5.5757240e+01f,
                5.6342510e+01f,  5.5779080e+01f,  5.6364560e+01f,  5.5800920e+01f,  5.6386610e+01f,  5.5822760e+01f,  5.6408660e+01f,  5.5844600e+01f,
                5.6430710e+01f,  5.5866440e+01f,  5.6452760e+01f,  5.5888280e+01f,  5.6474810e+01f,  5.5910120e+01f,  5.6540960e+01f,  5.5975640e+01f,
                5.6563010e+01f,  5.5997480e+01f,  5.6585060e+01f,  5.6019320e+01f,  5.6607110e+01f,  5.6041160e+01f,  5.6629160e+01f,  5.6063000e+01f,
                5.6651210e+01f,  5.6084840e+01f,  5.6673260e+01f,  5.6106680e+01f,  5.6695310e+01f,  5.6128520e+01f,  5.6717360e+01f,  5.6150360e+01f,
                5.6739410e+01f,  5.6172200e+01f,  5.6761460e+01f,  5.6194040e+01f,  5.6827610e+01f,  5.6259560e+01f,  5.6849660e+01f,  5.6281400e+01f,
                5.6871710e+01f,  5.6303240e+01f,  5.6893760e+01f,  5.6325080e+01f,  5.6915810e+01f,  5.6346920e+01f,  5.6937860e+01f,  5.6368760e+01f,
                5.6959910e+01f,  5.6390600e+01f,  5.6981960e+01f,  5.6412440e+01f,  5.7004010e+01f,  5.6434280e+01f,  5.7026060e+01f,  5.6456120e+01f,
                5.7048110e+01f,  5.6477960e+01f,  5.7114260e+01f,  5.6543480e+01f,  5.7136310e+01f,  5.6565320e+01f,  5.7158360e+01f,  5.6587160e+01f,
                5.7180410e+01f,  5.6609000e+01f,  5.7202460e+01f,  5.6630840e+01f,  5.7224510e+01f,  5.6652680e+01f,  5.7246560e+01f,  5.6674520e+01f,
                5.7268610e+01f,  5.6696360e+01f,  5.7290660e+01f,  5.6718200e+01f,  5.7312710e+01f,  5.6740040e+01f,  5.7334760e+01f,  5.6761880e+01f,
                5.8547510e+01f,  5.7963080e+01f,  5.8569560e+01f,  5.7984920e+01f,  5.8591610e+01f,  5.8006760e+01f,  5.8613660e+01f,  5.8028600e+01f,
                5.8635710e+01f,  5.8050440e+01f,  5.8657760e+01f,  5.8072280e+01f,  5.8679810e+01f,  5.8094120e+01f,  5.8701860e+01f,  5.8115960e+01f,
                5.8723910e+01f,  5.8137800e+01f,  5.8745960e+01f,  5.8159640e+01f,  5.8768010e+01f,  5.8181480e+01f,  5.8834160e+01f,  5.8247000e+01f,
                5.8856210e+01f,  5.8268840e+01f,  5.8878260e+01f,  5.8290680e+01f,  5.8900310e+01f,  5.8312520e+01f,  5.8922360e+01f,  5.8334360e+01f,
                5.8944410e+01f,  5.8356200e+01f,  5.8966460e+01f,  5.8378040e+01f,  5.8988510e+01f,  5.8399880e+01f,  5.9010560e+01f,  5.8421720e+01f,
                5.9032610e+01f,  5.8443560e+01f,  5.9054660e+01f,  5.8465400e+01f,  5.9120810e+01f,  5.8530920e+01f,  5.9142860e+01f,  5.8552760e+01f,
                5.9164910e+01f,  5.8574600e+01f,  5.9186960e+01f,  5.8596440e+01f,  5.9209010e+01f,  5.8618280e+01f,  5.9231060e+01f,  5.8640120e+01f,
                5.9253110e+01f,  5.8661960e+01f,  5.9275160e+01f,  5.8683800e+01f,  5.9297210e+01f,  5.8705640e+01f,  5.9319260e+01f,  5.8727480e+01f,
                5.9341310e+01f,  5.8749320e+01f,  5.9407460e+01f,  5.8814840e+01f,  5.9429510e+01f,  5.8836680e+01f,  5.9451560e+01f,  5.8858520e+01f,
                5.9473610e+01f,  5.8880360e+01f,  5.9495660e+01f,  5.8902200e+01f,  5.9517710e+01f,  5.8924040e+01f,  5.9539760e+01f,  5.8945880e+01f,
                5.9561810e+01f,  5.8967720e+01f,  5.9583860e+01f,  5.8989560e+01f,  5.9605910e+01f,  5.9011400e+01f,  5.9627960e+01f,  5.9033240e+01f,
                5.9694110e+01f,  5.9098760e+01f,  5.9716160e+01f,  5.9120600e+01f,  5.9738210e+01f,  5.9142440e+01f,  5.9760260e+01f,  5.9164280e+01f,
                5.9782310e+01f,  5.9186120e+01f,  5.9804360e+01f,  5.9207960e+01f,  5.9826410e+01f,  5.9229800e+01f,  5.9848460e+01f,  5.9251640e+01f,
                5.9870510e+01f,  5.9273480e+01f,  5.9892560e+01f,  5.9295320e+01f,  5.9914610e+01f,  5.9317160e+01f,  5.9980760e+01f,  5.9382680e+01f,
                6.0002810e+01f,  5.9404520e+01f,  6.0024860e+01f,  5.9426360e+01f,  6.0046910e+01f,  5.9448200e+01f,  6.0068960e+01f,  5.9470040e+01f,
                6.0091010e+01f,  5.9491880e+01f,  6.0113060e+01f,  5.9513720e+01f,  6.0135110e+01f,  5.9535560e+01f,  6.0157160e+01f,  5.9557400e+01f,
                6.0179210e+01f,  5.9579240e+01f,  6.0201260e+01f,  5.9601080e+01f,  6.0267410e+01f,  5.9666600e+01f,  6.0289460e+01f,  5.9688440e+01f,
                6.0311510e+01f,  5.9710280e+01f,  6.0333560e+01f,  5.9732120e+01f,  6.0355610e+01f,  5.9753960e+01f,  6.0377660e+01f,  5.9775800e+01f,
                6.0399710e+01f,  5.9797640e+01f,  6.0421760e+01f,  5.9819480e+01f,  6.0443810e+01f,  5.9841320e+01f,  6.0465860e+01f,  5.9863160e+01f,
                6.0487910e+01f,  5.9885000e+01f,  6.0554060e+01f,  5.9950520e+01f,  6.0576110e+01f,  5.9972360e+01f,  6.0598160e+01f,  5.9994200e+01f,
                6.0620210e+01f,  6.0016040e+01f,  6.0642260e+01f,  6.0037880e+01f,  6.0664310e+01f,  6.0059720e+01f,  6.0686360e+01f,  6.0081560e+01f,
                6.0708410e+01f,  6.0103400e+01f,  6.0730460e+01f,  6.0125240e+01f,  6.0752510e+01f,  6.0147080e+01f,  6.0774560e+01f,  6.0168920e+01f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {channels},{kwidth},{kheight},{inwidth},{inheight},{batch}");
        }
    }
}
