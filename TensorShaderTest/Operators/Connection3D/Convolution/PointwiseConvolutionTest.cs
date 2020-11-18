using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class PointwiseConvolutionTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        foreach ((int width, int height, int depth) in new (int, int, int)[] { (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {
                            float[] xval = (new float[width * height * depth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map3D x = new Map3D(inchannels, width, height, depth, batch, xval);
                            Filter3D w = new Filter3D(inchannels, outchannels, 1, 1, 1, wval);

                            Map3D y = Reference(x, w);

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, width, height, depth, batch), xval);
                            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels), wval);

                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, width, height, depth, batch));

                            PointwiseConvolution ope = new PointwiseConvolution(width, height, depth, inchannels, outchannels, batch);

                            ope.Execute(x_tensor, w_tensor, y_tensor);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{width},{height},{depth},{batch}");

                            Console.WriteLine($"pass: {inchannels},{outchannels},{width},{height},{depth},{batch}");
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
            int inchannels = 49, outchannels = 50;
            int width = 128, height = 196, depth = 4;

            float[] xval = (new float[width * height * depth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map3D x = new Map3D(inchannels, width, height, depth, batch, xval);
            Filter3D w = new Filter3D(inchannels, outchannels, 1, 1, 1, wval);

            Map3D y = Reference(x, w);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, width, height, depth, batch), xval);
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels), wval);

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, width, height, depth, batch));

            PointwiseConvolution ope = new PointwiseConvolution(width, height, depth, inchannels, outchannels, batch);

            ope.Execute(x_tensor, w_tensor, y_tensor);

            float[] y_expect = y.ToArray();
            float[] y_actual = y_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{width},{height},{depth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{width},{height},{depth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int width = 64, height = 64, depth = 64, inchannels = 31, outchannels = 31;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, width, height, depth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, width, height, depth));

            PointwiseConvolution ope = new PointwiseConvolution(width, height, depth, inchannels, outchannels);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/ptwise_convolution_3d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map3D Reference(Map3D x, Filter3D w) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth;

            Map3D y = new Map3D(outchannels, inw, inh, ind, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy, oz = 0; oz < ind; oz++) {
                    for (oy = 0; oy < inh; oy++) {
                        for (ox = 0; ox < inw; ox++) {
                            for (int outch = 0; outch < outchannels; outch++) {
                                double sum = y[outch, ox, oy, oz, th];

                                for (int inch = 0; inch < inchannels; inch++) {
                                    sum += x[inch, ox, oy, oz, th] * w[inch, outch, 0, 0, 0];
                                }

                                y[outch, ox, oy, oz, th] = sum;
                            }
                        }
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 2, outchannels = 3, inwidth = 7, inheight = 6, indepth = 5, batch = 2;

            float[] xval = (new float[batch * inwidth * inheight * indepth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map3D x = new Map3D(inchannels, inwidth, inheight, indepth, batch, xval);
            Filter3D w = new Filter3D(inchannels, outchannels, 1, 1, 1, wval);

            Map3D y = Reference(x, w);

            float[] y_expect = {
                4.0000e-06f,  2.0000e-06f,  0.0000e+00f,  2.2000e-05f,  1.2000e-05f,  2.0000e-06f,  4.0000e-05f,  2.2000e-05f,  4.0000e-06f,  5.8000e-05f,  3.2000e-05f,  6.0000e-06f,
                7.6000e-05f,  4.2000e-05f,  8.0000e-06f,  9.4000e-05f,  5.2000e-05f,  1.0000e-05f,  1.1200e-04f,  6.2000e-05f,  1.2000e-05f,  1.3000e-04f,  7.2000e-05f,  1.4000e-05f,
                1.4800e-04f,  8.2000e-05f,  1.6000e-05f,  1.6600e-04f,  9.2000e-05f,  1.8000e-05f,  1.8400e-04f,  1.0200e-04f,  2.0000e-05f,  2.0200e-04f,  1.1200e-04f,  2.2000e-05f,
                2.2000e-04f,  1.2200e-04f,  2.4000e-05f,  2.3800e-04f,  1.3200e-04f,  2.6000e-05f,  2.5600e-04f,  1.4200e-04f,  2.8000e-05f,  2.7400e-04f,  1.5200e-04f,  3.0000e-05f,
                2.9200e-04f,  1.6200e-04f,  3.2000e-05f,  3.1000e-04f,  1.7200e-04f,  3.4000e-05f,  3.2800e-04f,  1.8200e-04f,  3.6000e-05f,  3.4600e-04f,  1.9200e-04f,  3.8000e-05f,
                3.6400e-04f,  2.0200e-04f,  4.0000e-05f,  3.8200e-04f,  2.1200e-04f,  4.2000e-05f,  4.0000e-04f,  2.2200e-04f,  4.4000e-05f,  4.1800e-04f,  2.3200e-04f,  4.6000e-05f,
                4.3600e-04f,  2.4200e-04f,  4.8000e-05f,  4.5400e-04f,  2.5200e-04f,  5.0000e-05f,  4.7200e-04f,  2.6200e-04f,  5.2000e-05f,  4.9000e-04f,  2.7200e-04f,  5.4000e-05f,
                5.0800e-04f,  2.8200e-04f,  5.6000e-05f,  5.2600e-04f,  2.9200e-04f,  5.8000e-05f,  5.4400e-04f,  3.0200e-04f,  6.0000e-05f,  5.6200e-04f,  3.1200e-04f,  6.2000e-05f,
                5.8000e-04f,  3.2200e-04f,  6.4000e-05f,  5.9800e-04f,  3.3200e-04f,  6.6000e-05f,  6.1600e-04f,  3.4200e-04f,  6.8000e-05f,  6.3400e-04f,  3.5200e-04f,  7.0000e-05f,
                6.5200e-04f,  3.6200e-04f,  7.2000e-05f,  6.7000e-04f,  3.7200e-04f,  7.4000e-05f,  6.8800e-04f,  3.8200e-04f,  7.6000e-05f,  7.0600e-04f,  3.9200e-04f,  7.8000e-05f,
                7.2400e-04f,  4.0200e-04f,  8.0000e-05f,  7.4200e-04f,  4.1200e-04f,  8.2000e-05f,  7.6000e-04f,  4.2200e-04f,  8.4000e-05f,  7.7800e-04f,  4.3200e-04f,  8.6000e-05f,
                7.9600e-04f,  4.4200e-04f,  8.8000e-05f,  8.1400e-04f,  4.5200e-04f,  9.0000e-05f,  8.3200e-04f,  4.6200e-04f,  9.2000e-05f,  8.5000e-04f,  4.7200e-04f,  9.4000e-05f,
                8.6800e-04f,  4.8200e-04f,  9.6000e-05f,  8.8600e-04f,  4.9200e-04f,  9.8000e-05f,  9.0400e-04f,  5.0200e-04f,  1.0000e-04f,  9.2200e-04f,  5.1200e-04f,  1.0200e-04f,
                9.4000e-04f,  5.2200e-04f,  1.0400e-04f,  9.5800e-04f,  5.3200e-04f,  1.0600e-04f,  9.7600e-04f,  5.4200e-04f,  1.0800e-04f,  9.9400e-04f,  5.5200e-04f,  1.1000e-04f,
                1.0120e-03f,  5.6200e-04f,  1.1200e-04f,  1.0300e-03f,  5.7200e-04f,  1.1400e-04f,  1.0480e-03f,  5.8200e-04f,  1.1600e-04f,  1.0660e-03f,  5.9200e-04f,  1.1800e-04f,
                1.0840e-03f,  6.0200e-04f,  1.2000e-04f,  1.1020e-03f,  6.1200e-04f,  1.2200e-04f,  1.1200e-03f,  6.2200e-04f,  1.2400e-04f,  1.1380e-03f,  6.3200e-04f,  1.2600e-04f,
                1.1560e-03f,  6.4200e-04f,  1.2800e-04f,  1.1740e-03f,  6.5200e-04f,  1.3000e-04f,  1.1920e-03f,  6.6200e-04f,  1.3200e-04f,  1.2100e-03f,  6.7200e-04f,  1.3400e-04f,
                1.2280e-03f,  6.8200e-04f,  1.3600e-04f,  1.2460e-03f,  6.9200e-04f,  1.3800e-04f,  1.2640e-03f,  7.0200e-04f,  1.4000e-04f,  1.2820e-03f,  7.1200e-04f,  1.4200e-04f,
                1.3000e-03f,  7.2200e-04f,  1.4400e-04f,  1.3180e-03f,  7.3200e-04f,  1.4600e-04f,  1.3360e-03f,  7.4200e-04f,  1.4800e-04f,  1.3540e-03f,  7.5200e-04f,  1.5000e-04f,
                1.3720e-03f,  7.6200e-04f,  1.5200e-04f,  1.3900e-03f,  7.7200e-04f,  1.5400e-04f,  1.4080e-03f,  7.8200e-04f,  1.5600e-04f,  1.4260e-03f,  7.9200e-04f,  1.5800e-04f,
                1.4440e-03f,  8.0200e-04f,  1.6000e-04f,  1.4620e-03f,  8.1200e-04f,  1.6200e-04f,  1.4800e-03f,  8.2200e-04f,  1.6400e-04f,  1.4980e-03f,  8.3200e-04f,  1.6600e-04f,
                1.5160e-03f,  8.4200e-04f,  1.6800e-04f,  1.5340e-03f,  8.5200e-04f,  1.7000e-04f,  1.5520e-03f,  8.6200e-04f,  1.7200e-04f,  1.5700e-03f,  8.7200e-04f,  1.7400e-04f,
                1.5880e-03f,  8.8200e-04f,  1.7600e-04f,  1.6060e-03f,  8.9200e-04f,  1.7800e-04f,  1.6240e-03f,  9.0200e-04f,  1.8000e-04f,  1.6420e-03f,  9.1200e-04f,  1.8200e-04f,
                1.6600e-03f,  9.2200e-04f,  1.8400e-04f,  1.6780e-03f,  9.3200e-04f,  1.8600e-04f,  1.6960e-03f,  9.4200e-04f,  1.8800e-04f,  1.7140e-03f,  9.5200e-04f,  1.9000e-04f,
                1.7320e-03f,  9.6200e-04f,  1.9200e-04f,  1.7500e-03f,  9.7200e-04f,  1.9400e-04f,  1.7680e-03f,  9.8200e-04f,  1.9600e-04f,  1.7860e-03f,  9.9200e-04f,  1.9800e-04f,
                1.8040e-03f,  1.0020e-03f,  2.0000e-04f,  1.8220e-03f,  1.0120e-03f,  2.0200e-04f,  1.8400e-03f,  1.0220e-03f,  2.0400e-04f,  1.8580e-03f,  1.0320e-03f,  2.0600e-04f,
                1.8760e-03f,  1.0420e-03f,  2.0800e-04f,  1.8940e-03f,  1.0520e-03f,  2.1000e-04f,  1.9120e-03f,  1.0620e-03f,  2.1200e-04f,  1.9300e-03f,  1.0720e-03f,  2.1400e-04f,
                1.9480e-03f,  1.0820e-03f,  2.1600e-04f,  1.9660e-03f,  1.0920e-03f,  2.1800e-04f,  1.9840e-03f,  1.1020e-03f,  2.2000e-04f,  2.0020e-03f,  1.1120e-03f,  2.2200e-04f,
                2.0200e-03f,  1.1220e-03f,  2.2400e-04f,  2.0380e-03f,  1.1320e-03f,  2.2600e-04f,  2.0560e-03f,  1.1420e-03f,  2.2800e-04f,  2.0740e-03f,  1.1520e-03f,  2.3000e-04f,
                2.0920e-03f,  1.1620e-03f,  2.3200e-04f,  2.1100e-03f,  1.1720e-03f,  2.3400e-04f,  2.1280e-03f,  1.1820e-03f,  2.3600e-04f,  2.1460e-03f,  1.1920e-03f,  2.3800e-04f,
                2.1640e-03f,  1.2020e-03f,  2.4000e-04f,  2.1820e-03f,  1.2120e-03f,  2.4200e-04f,  2.2000e-03f,  1.2220e-03f,  2.4400e-04f,  2.2180e-03f,  1.2320e-03f,  2.4600e-04f,
                2.2360e-03f,  1.2420e-03f,  2.4800e-04f,  2.2540e-03f,  1.2520e-03f,  2.5000e-04f,  2.2720e-03f,  1.2620e-03f,  2.5200e-04f,  2.2900e-03f,  1.2720e-03f,  2.5400e-04f,
                2.3080e-03f,  1.2820e-03f,  2.5600e-04f,  2.3260e-03f,  1.2920e-03f,  2.5800e-04f,  2.3440e-03f,  1.3020e-03f,  2.6000e-04f,  2.3620e-03f,  1.3120e-03f,  2.6200e-04f,
                2.3800e-03f,  1.3220e-03f,  2.6400e-04f,  2.3980e-03f,  1.3320e-03f,  2.6600e-04f,  2.4160e-03f,  1.3420e-03f,  2.6800e-04f,  2.4340e-03f,  1.3520e-03f,  2.7000e-04f,
                2.4520e-03f,  1.3620e-03f,  2.7200e-04f,  2.4700e-03f,  1.3720e-03f,  2.7400e-04f,  2.4880e-03f,  1.3820e-03f,  2.7600e-04f,  2.5060e-03f,  1.3920e-03f,  2.7800e-04f,
                2.5240e-03f,  1.4020e-03f,  2.8000e-04f,  2.5420e-03f,  1.4120e-03f,  2.8200e-04f,  2.5600e-03f,  1.4220e-03f,  2.8400e-04f,  2.5780e-03f,  1.4320e-03f,  2.8600e-04f,
                2.5960e-03f,  1.4420e-03f,  2.8800e-04f,  2.6140e-03f,  1.4520e-03f,  2.9000e-04f,  2.6320e-03f,  1.4620e-03f,  2.9200e-04f,  2.6500e-03f,  1.4720e-03f,  2.9400e-04f,
                2.6680e-03f,  1.4820e-03f,  2.9600e-04f,  2.6860e-03f,  1.4920e-03f,  2.9800e-04f,  2.7040e-03f,  1.5020e-03f,  3.0000e-04f,  2.7220e-03f,  1.5120e-03f,  3.0200e-04f,
                2.7400e-03f,  1.5220e-03f,  3.0400e-04f,  2.7580e-03f,  1.5320e-03f,  3.0600e-04f,  2.7760e-03f,  1.5420e-03f,  3.0800e-04f,  2.7940e-03f,  1.5520e-03f,  3.1000e-04f,
                2.8120e-03f,  1.5620e-03f,  3.1200e-04f,  2.8300e-03f,  1.5720e-03f,  3.1400e-04f,  2.8480e-03f,  1.5820e-03f,  3.1600e-04f,  2.8660e-03f,  1.5920e-03f,  3.1800e-04f,
                2.8840e-03f,  1.6020e-03f,  3.2000e-04f,  2.9020e-03f,  1.6120e-03f,  3.2200e-04f,  2.9200e-03f,  1.6220e-03f,  3.2400e-04f,  2.9380e-03f,  1.6320e-03f,  3.2600e-04f,
                2.9560e-03f,  1.6420e-03f,  3.2800e-04f,  2.9740e-03f,  1.6520e-03f,  3.3000e-04f,  2.9920e-03f,  1.6620e-03f,  3.3200e-04f,  3.0100e-03f,  1.6720e-03f,  3.3400e-04f,
                3.0280e-03f,  1.6820e-03f,  3.3600e-04f,  3.0460e-03f,  1.6920e-03f,  3.3800e-04f,  3.0640e-03f,  1.7020e-03f,  3.4000e-04f,  3.0820e-03f,  1.7120e-03f,  3.4200e-04f,
                3.1000e-03f,  1.7220e-03f,  3.4400e-04f,  3.1180e-03f,  1.7320e-03f,  3.4600e-04f,  3.1360e-03f,  1.7420e-03f,  3.4800e-04f,  3.1540e-03f,  1.7520e-03f,  3.5000e-04f,
                3.1720e-03f,  1.7620e-03f,  3.5200e-04f,  3.1900e-03f,  1.7720e-03f,  3.5400e-04f,  3.2080e-03f,  1.7820e-03f,  3.5600e-04f,  3.2260e-03f,  1.7920e-03f,  3.5800e-04f,
                3.2440e-03f,  1.8020e-03f,  3.6000e-04f,  3.2620e-03f,  1.8120e-03f,  3.6200e-04f,  3.2800e-03f,  1.8220e-03f,  3.6400e-04f,  3.2980e-03f,  1.8320e-03f,  3.6600e-04f,
                3.3160e-03f,  1.8420e-03f,  3.6800e-04f,  3.3340e-03f,  1.8520e-03f,  3.7000e-04f,  3.3520e-03f,  1.8620e-03f,  3.7200e-04f,  3.3700e-03f,  1.8720e-03f,  3.7400e-04f,
                3.3880e-03f,  1.8820e-03f,  3.7600e-04f,  3.4060e-03f,  1.8920e-03f,  3.7800e-04f,  3.4240e-03f,  1.9020e-03f,  3.8000e-04f,  3.4420e-03f,  1.9120e-03f,  3.8200e-04f,
                3.4600e-03f,  1.9220e-03f,  3.8400e-04f,  3.4780e-03f,  1.9320e-03f,  3.8600e-04f,  3.4960e-03f,  1.9420e-03f,  3.8800e-04f,  3.5140e-03f,  1.9520e-03f,  3.9000e-04f,
                3.5320e-03f,  1.9620e-03f,  3.9200e-04f,  3.5500e-03f,  1.9720e-03f,  3.9400e-04f,  3.5680e-03f,  1.9820e-03f,  3.9600e-04f,  3.5860e-03f,  1.9920e-03f,  3.9800e-04f,
                3.6040e-03f,  2.0020e-03f,  4.0000e-04f,  3.6220e-03f,  2.0120e-03f,  4.0200e-04f,  3.6400e-03f,  2.0220e-03f,  4.0400e-04f,  3.6580e-03f,  2.0320e-03f,  4.0600e-04f,
                3.6760e-03f,  2.0420e-03f,  4.0800e-04f,  3.6940e-03f,  2.0520e-03f,  4.1000e-04f,  3.7120e-03f,  2.0620e-03f,  4.1200e-04f,  3.7300e-03f,  2.0720e-03f,  4.1400e-04f,
                3.7480e-03f,  2.0820e-03f,  4.1600e-04f,  3.7660e-03f,  2.0920e-03f,  4.1800e-04f,  3.7840e-03f,  2.1020e-03f,  4.2000e-04f,  3.8020e-03f,  2.1120e-03f,  4.2200e-04f,
                3.8200e-03f,  2.1220e-03f,  4.2400e-04f,  3.8380e-03f,  2.1320e-03f,  4.2600e-04f,  3.8560e-03f,  2.1420e-03f,  4.2800e-04f,  3.8740e-03f,  2.1520e-03f,  4.3000e-04f,
                3.8920e-03f,  2.1620e-03f,  4.3200e-04f,  3.9100e-03f,  2.1720e-03f,  4.3400e-04f,  3.9280e-03f,  2.1820e-03f,  4.3600e-04f,  3.9460e-03f,  2.1920e-03f,  4.3800e-04f,
                3.9640e-03f,  2.2020e-03f,  4.4000e-04f,  3.9820e-03f,  2.2120e-03f,  4.4200e-04f,  4.0000e-03f,  2.2220e-03f,  4.4400e-04f,  4.0180e-03f,  2.2320e-03f,  4.4600e-04f,
                4.0360e-03f,  2.2420e-03f,  4.4800e-04f,  4.0540e-03f,  2.2520e-03f,  4.5000e-04f,  4.0720e-03f,  2.2620e-03f,  4.5200e-04f,  4.0900e-03f,  2.2720e-03f,  4.5400e-04f,
                4.1080e-03f,  2.2820e-03f,  4.5600e-04f,  4.1260e-03f,  2.2920e-03f,  4.5800e-04f,  4.1440e-03f,  2.3020e-03f,  4.6000e-04f,  4.1620e-03f,  2.3120e-03f,  4.6200e-04f,
                4.1800e-03f,  2.3220e-03f,  4.6400e-04f,  4.1980e-03f,  2.3320e-03f,  4.6600e-04f,  4.2160e-03f,  2.3420e-03f,  4.6800e-04f,  4.2340e-03f,  2.3520e-03f,  4.7000e-04f,
                4.2520e-03f,  2.3620e-03f,  4.7200e-04f,  4.2700e-03f,  2.3720e-03f,  4.7400e-04f,  4.2880e-03f,  2.3820e-03f,  4.7600e-04f,  4.3060e-03f,  2.3920e-03f,  4.7800e-04f,
                4.3240e-03f,  2.4020e-03f,  4.8000e-04f,  4.3420e-03f,  2.4120e-03f,  4.8200e-04f,  4.3600e-03f,  2.4220e-03f,  4.8400e-04f,  4.3780e-03f,  2.4320e-03f,  4.8600e-04f,
                4.3960e-03f,  2.4420e-03f,  4.8800e-04f,  4.4140e-03f,  2.4520e-03f,  4.9000e-04f,  4.4320e-03f,  2.4620e-03f,  4.9200e-04f,  4.4500e-03f,  2.4720e-03f,  4.9400e-04f,
                4.4680e-03f,  2.4820e-03f,  4.9600e-04f,  4.4860e-03f,  2.4920e-03f,  4.9800e-04f,  4.5040e-03f,  2.5020e-03f,  5.0000e-04f,  4.5220e-03f,  2.5120e-03f,  5.0200e-04f,
                4.5400e-03f,  2.5220e-03f,  5.0400e-04f,  4.5580e-03f,  2.5320e-03f,  5.0600e-04f,  4.5760e-03f,  2.5420e-03f,  5.0800e-04f,  4.5940e-03f,  2.5520e-03f,  5.1000e-04f,
                4.6120e-03f,  2.5620e-03f,  5.1200e-04f,  4.6300e-03f,  2.5720e-03f,  5.1400e-04f,  4.6480e-03f,  2.5820e-03f,  5.1600e-04f,  4.6660e-03f,  2.5920e-03f,  5.1800e-04f,
                4.6840e-03f,  2.6020e-03f,  5.2000e-04f,  4.7020e-03f,  2.6120e-03f,  5.2200e-04f,  4.7200e-03f,  2.6220e-03f,  5.2400e-04f,  4.7380e-03f,  2.6320e-03f,  5.2600e-04f,
                4.7560e-03f,  2.6420e-03f,  5.2800e-04f,  4.7740e-03f,  2.6520e-03f,  5.3000e-04f,  4.7920e-03f,  2.6620e-03f,  5.3200e-04f,  4.8100e-03f,  2.6720e-03f,  5.3400e-04f,
                4.8280e-03f,  2.6820e-03f,  5.3600e-04f,  4.8460e-03f,  2.6920e-03f,  5.3800e-04f,  4.8640e-03f,  2.7020e-03f,  5.4000e-04f,  4.8820e-03f,  2.7120e-03f,  5.4200e-04f,
                4.9000e-03f,  2.7220e-03f,  5.4400e-04f,  4.9180e-03f,  2.7320e-03f,  5.4600e-04f,  4.9360e-03f,  2.7420e-03f,  5.4800e-04f,  4.9540e-03f,  2.7520e-03f,  5.5000e-04f,
                4.9720e-03f,  2.7620e-03f,  5.5200e-04f,  4.9900e-03f,  2.7720e-03f,  5.5400e-04f,  5.0080e-03f,  2.7820e-03f,  5.5600e-04f,  5.0260e-03f,  2.7920e-03f,  5.5800e-04f,
                5.0440e-03f,  2.8020e-03f,  5.6000e-04f,  5.0620e-03f,  2.8120e-03f,  5.6200e-04f,  5.0800e-03f,  2.8220e-03f,  5.6400e-04f,  5.0980e-03f,  2.8320e-03f,  5.6600e-04f,
                5.1160e-03f,  2.8420e-03f,  5.6800e-04f,  5.1340e-03f,  2.8520e-03f,  5.7000e-04f,  5.1520e-03f,  2.8620e-03f,  5.7200e-04f,  5.1700e-03f,  2.8720e-03f,  5.7400e-04f,
                5.1880e-03f,  2.8820e-03f,  5.7600e-04f,  5.2060e-03f,  2.8920e-03f,  5.7800e-04f,  5.2240e-03f,  2.9020e-03f,  5.8000e-04f,  5.2420e-03f,  2.9120e-03f,  5.8200e-04f,
                5.2600e-03f,  2.9220e-03f,  5.8400e-04f,  5.2780e-03f,  2.9320e-03f,  5.8600e-04f,  5.2960e-03f,  2.9420e-03f,  5.8800e-04f,  5.3140e-03f,  2.9520e-03f,  5.9000e-04f,
                5.3320e-03f,  2.9620e-03f,  5.9200e-04f,  5.3500e-03f,  2.9720e-03f,  5.9400e-04f,  5.3680e-03f,  2.9820e-03f,  5.9600e-04f,  5.3860e-03f,  2.9920e-03f,  5.9800e-04f,
                5.4040e-03f,  3.0020e-03f,  6.0000e-04f,  5.4220e-03f,  3.0120e-03f,  6.0200e-04f,  5.4400e-03f,  3.0220e-03f,  6.0400e-04f,  5.4580e-03f,  3.0320e-03f,  6.0600e-04f,
                5.4760e-03f,  3.0420e-03f,  6.0800e-04f,  5.4940e-03f,  3.0520e-03f,  6.1000e-04f,  5.5120e-03f,  3.0620e-03f,  6.1200e-04f,  5.5300e-03f,  3.0720e-03f,  6.1400e-04f,
                5.5480e-03f,  3.0820e-03f,  6.1600e-04f,  5.5660e-03f,  3.0920e-03f,  6.1800e-04f,  5.5840e-03f,  3.1020e-03f,  6.2000e-04f,  5.6020e-03f,  3.1120e-03f,  6.2200e-04f,
                5.6200e-03f,  3.1220e-03f,  6.2400e-04f,  5.6380e-03f,  3.1320e-03f,  6.2600e-04f,  5.6560e-03f,  3.1420e-03f,  6.2800e-04f,  5.6740e-03f,  3.1520e-03f,  6.3000e-04f,
                5.6920e-03f,  3.1620e-03f,  6.3200e-04f,  5.7100e-03f,  3.1720e-03f,  6.3400e-04f,  5.7280e-03f,  3.1820e-03f,  6.3600e-04f,  5.7460e-03f,  3.1920e-03f,  6.3800e-04f,
                5.7640e-03f,  3.2020e-03f,  6.4000e-04f,  5.7820e-03f,  3.2120e-03f,  6.4200e-04f,  5.8000e-03f,  3.2220e-03f,  6.4400e-04f,  5.8180e-03f,  3.2320e-03f,  6.4600e-04f,
                5.8360e-03f,  3.2420e-03f,  6.4800e-04f,  5.8540e-03f,  3.2520e-03f,  6.5000e-04f,  5.8720e-03f,  3.2620e-03f,  6.5200e-04f,  5.8900e-03f,  3.2720e-03f,  6.5400e-04f,
                5.9080e-03f,  3.2820e-03f,  6.5600e-04f,  5.9260e-03f,  3.2920e-03f,  6.5800e-04f,  5.9440e-03f,  3.3020e-03f,  6.6000e-04f,  5.9620e-03f,  3.3120e-03f,  6.6200e-04f,
                5.9800e-03f,  3.3220e-03f,  6.6400e-04f,  5.9980e-03f,  3.3320e-03f,  6.6600e-04f,  6.0160e-03f,  3.3420e-03f,  6.6800e-04f,  6.0340e-03f,  3.3520e-03f,  6.7000e-04f,
                6.0520e-03f,  3.3620e-03f,  6.7200e-04f,  6.0700e-03f,  3.3720e-03f,  6.7400e-04f,  6.0880e-03f,  3.3820e-03f,  6.7600e-04f,  6.1060e-03f,  3.3920e-03f,  6.7800e-04f,
                6.1240e-03f,  3.4020e-03f,  6.8000e-04f,  6.1420e-03f,  3.4120e-03f,  6.8200e-04f,  6.1600e-03f,  3.4220e-03f,  6.8400e-04f,  6.1780e-03f,  3.4320e-03f,  6.8600e-04f,
                6.1960e-03f,  3.4420e-03f,  6.8800e-04f,  6.2140e-03f,  3.4520e-03f,  6.9000e-04f,  6.2320e-03f,  3.4620e-03f,  6.9200e-04f,  6.2500e-03f,  3.4720e-03f,  6.9400e-04f,
                6.2680e-03f,  3.4820e-03f,  6.9600e-04f,  6.2860e-03f,  3.4920e-03f,  6.9800e-04f,  6.3040e-03f,  3.5020e-03f,  7.0000e-04f,  6.3220e-03f,  3.5120e-03f,  7.0200e-04f,
                6.3400e-03f,  3.5220e-03f,  7.0400e-04f,  6.3580e-03f,  3.5320e-03f,  7.0600e-04f,  6.3760e-03f,  3.5420e-03f,  7.0800e-04f,  6.3940e-03f,  3.5520e-03f,  7.1000e-04f,
                6.4120e-03f,  3.5620e-03f,  7.1200e-04f,  6.4300e-03f,  3.5720e-03f,  7.1400e-04f,  6.4480e-03f,  3.5820e-03f,  7.1600e-04f,  6.4660e-03f,  3.5920e-03f,  7.1800e-04f,
                6.4840e-03f,  3.6020e-03f,  7.2000e-04f,  6.5020e-03f,  3.6120e-03f,  7.2200e-04f,  6.5200e-03f,  3.6220e-03f,  7.2400e-04f,  6.5380e-03f,  3.6320e-03f,  7.2600e-04f,
                6.5560e-03f,  3.6420e-03f,  7.2800e-04f,  6.5740e-03f,  3.6520e-03f,  7.3000e-04f,  6.5920e-03f,  3.6620e-03f,  7.3200e-04f,  6.6100e-03f,  3.6720e-03f,  7.3400e-04f,
                6.6280e-03f,  3.6820e-03f,  7.3600e-04f,  6.6460e-03f,  3.6920e-03f,  7.3800e-04f,  6.6640e-03f,  3.7020e-03f,  7.4000e-04f,  6.6820e-03f,  3.7120e-03f,  7.4200e-04f,
                6.7000e-03f,  3.7220e-03f,  7.4400e-04f,  6.7180e-03f,  3.7320e-03f,  7.4600e-04f,  6.7360e-03f,  3.7420e-03f,  7.4800e-04f,  6.7540e-03f,  3.7520e-03f,  7.5000e-04f,
                6.7720e-03f,  3.7620e-03f,  7.5200e-04f,  6.7900e-03f,  3.7720e-03f,  7.5400e-04f,  6.8080e-03f,  3.7820e-03f,  7.5600e-04f,  6.8260e-03f,  3.7920e-03f,  7.5800e-04f,
                6.8440e-03f,  3.8020e-03f,  7.6000e-04f,  6.8620e-03f,  3.8120e-03f,  7.6200e-04f,  6.8800e-03f,  3.8220e-03f,  7.6400e-04f,  6.8980e-03f,  3.8320e-03f,  7.6600e-04f,
                6.9160e-03f,  3.8420e-03f,  7.6800e-04f,  6.9340e-03f,  3.8520e-03f,  7.7000e-04f,  6.9520e-03f,  3.8620e-03f,  7.7200e-04f,  6.9700e-03f,  3.8720e-03f,  7.7400e-04f,
                6.9880e-03f,  3.8820e-03f,  7.7600e-04f,  7.0060e-03f,  3.8920e-03f,  7.7800e-04f,  7.0240e-03f,  3.9020e-03f,  7.8000e-04f,  7.0420e-03f,  3.9120e-03f,  7.8200e-04f,
                7.0600e-03f,  3.9220e-03f,  7.8400e-04f,  7.0780e-03f,  3.9320e-03f,  7.8600e-04f,  7.0960e-03f,  3.9420e-03f,  7.8800e-04f,  7.1140e-03f,  3.9520e-03f,  7.9000e-04f,
                7.1320e-03f,  3.9620e-03f,  7.9200e-04f,  7.1500e-03f,  3.9720e-03f,  7.9400e-04f,  7.1680e-03f,  3.9820e-03f,  7.9600e-04f,  7.1860e-03f,  3.9920e-03f,  7.9800e-04f,
                7.2040e-03f,  4.0020e-03f,  8.0000e-04f,  7.2220e-03f,  4.0120e-03f,  8.0200e-04f,  7.2400e-03f,  4.0220e-03f,  8.0400e-04f,  7.2580e-03f,  4.0320e-03f,  8.0600e-04f,
                7.2760e-03f,  4.0420e-03f,  8.0800e-04f,  7.2940e-03f,  4.0520e-03f,  8.1000e-04f,  7.3120e-03f,  4.0620e-03f,  8.1200e-04f,  7.3300e-03f,  4.0720e-03f,  8.1400e-04f,
                7.3480e-03f,  4.0820e-03f,  8.1600e-04f,  7.3660e-03f,  4.0920e-03f,  8.1800e-04f,  7.3840e-03f,  4.1020e-03f,  8.2000e-04f,  7.4020e-03f,  4.1120e-03f,  8.2200e-04f,
                7.4200e-03f,  4.1220e-03f,  8.2400e-04f,  7.4380e-03f,  4.1320e-03f,  8.2600e-04f,  7.4560e-03f,  4.1420e-03f,  8.2800e-04f,  7.4740e-03f,  4.1520e-03f,  8.3000e-04f,
                7.4920e-03f,  4.1620e-03f,  8.3200e-04f,  7.5100e-03f,  4.1720e-03f,  8.3400e-04f,  7.5280e-03f,  4.1820e-03f,  8.3600e-04f,  7.5460e-03f,  4.1920e-03f,  8.3800e-04f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{inwidth},{inheight},{indepth},{batch}");
        }
    }
}
