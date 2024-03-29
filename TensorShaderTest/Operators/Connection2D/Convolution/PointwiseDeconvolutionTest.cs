using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class PointwiseDeconvolutionTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        foreach (int height in new int[] { 8, 9, 19, 23 }) {
                            foreach (int width in new int[] { 8, 9, 13, 17 }) {
                                float[] yval = (new float[width * height * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                Map2D y = new(outchannels, width, height, batch, yval);
                                Filter2D w = new(inchannels, outchannels, 1, 1, wval);

                                Map2D x = Reference(y, w);

                                OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, width, height, batch), yval);
                                OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

                                OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, width, height, batch));

                                PointwiseDeconvolution ope = new(width, height, outchannels, inchannels, batch);

                                ope.Execute(y_tensor, w_tensor, x_tensor);

                                float[] x_expect = x.ToArray();
                                float[] x_actual = x_tensor.State.Value;

                                CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                                CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                AssertError.Tolerance(x_expect, x_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value{inchannels},{outchannels},{width},{height},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{width},{height},{batch}");
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
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        foreach (int height in new int[] { 8, 9, 19, 23 }) {
                            foreach (int width in new int[] { 8, 9, 13, 17 }) {
                                float[] yval = (new float[width * height * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                Map2D y = new(outchannels, width, height, batch, yval);
                                Filter2D w = new(inchannels, outchannels, 1, 1, wval);

                                Map2D x = Reference(y, w);

                                OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, width, height, batch), yval);
                                OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

                                OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, width, height, batch));

                                PointwiseDeconvolution ope = new(width, height, outchannels, inchannels, batch);

                                ope.Execute(y_tensor, w_tensor, x_tensor);

                                float[] x_expect = x.ToArray();
                                float[] x_actual = x_tensor.State.Value;

                                CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                                CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value{inchannels},{outchannels},{width},{height},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{width},{height},{batch}");
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void ExecuteCudnnTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = true;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        foreach (int height in new int[] { 8, 9, 19, 23 }) {
                            foreach (int width in new int[] { 8, 9, 13, 17 }) {
                                float[] yval = (new float[width * height * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                Map2D y = new(outchannels, width, height, batch, yval);
                                Filter2D w = new(inchannels, outchannels, 1, 1, wval);

                                Map2D x = Reference(y, w);

                                OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, width, height, batch), yval);
                                OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

                                OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, width, height, batch));

                                PointwiseDeconvolution ope = new(width, height, outchannels, inchannels, batch);

                                ope.Execute(y_tensor, w_tensor, x_tensor);

                                float[] x_expect = x.ToArray();
                                float[] x_actual = x_tensor.State.Value;

                                CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                                CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                AssertError.Tolerance(x_expect, x_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value{inchannels},{outchannels},{width},{height},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{width},{height},{batch}");
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
            int inchannels = 49, outchannels = 50;
            int width = 128, height = 196;

            float[] yval = (new float[width * height * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map2D y = new(outchannels, width, height, batch, yval);
            Filter2D w = new(inchannels, outchannels, 1, 1, wval);

            Map2D x = Reference(y, w);

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, width, height, batch), yval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, width, height, batch));

            PointwiseDeconvolution ope = new(width, height, outchannels, inchannels, batch);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            float[] x_expect = x.ToArray();
            float[] x_actual = x_tensor.State.Value;

            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value{inchannels},{outchannels},{width},{height},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{width},{height},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 512, inheight = 512, inchannels = 31, outchannels = 63;

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, inwidth, inheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));

            PointwiseDeconvolution ope = new(inwidth, inheight, outchannels, inchannels);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/ptwise_deconvolution_2d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 512, inheight = 512, inchannels = 31, outchannels = 63;

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, inwidth, inheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));

            PointwiseDeconvolution ope = new(inwidth, inheight, outchannels, inchannels);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/ptwise_deconvolution_2d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedCudnnTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = true;

            int inwidth = 512, inheight = 512, inchannels = 31, outchannels = 63;

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, inwidth, inheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));

            PointwiseDeconvolution ope = new(inwidth, inheight, outchannels, inchannels);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/ptwise_deconvolution_2d_cudnn.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map2D Reference(Map2D y, Filter2D w) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int inw = y.Width, inh = y.Height;

            Map2D x = new(inchannels, inw, inh, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix, iy = 0; iy < inh; iy++) {
                    for (ix = 0; ix < inw; ix++) {
                        for (int outch = 0; outch < outchannels; outch++) {
                            double v = y[outch, ix, iy, th];

                            for (int inch = 0; inch < inchannels; inch++) {
                                x[inch, ix, iy, th] += v * w[inch, outch, 0, 0];
                            }
                        }
                    }
                }
            }

            return x;
        }

        public static Map2D Reference2(Map2D y, Filter2D w) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int inw = y.Width, inh = y.Height;

            Map2D x = new(inchannels, inw, inh, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix, iy = 0; iy < inh; iy++) {
                    for (ix = 0; ix < inw; ix++) {
                        int inch;
                        double[] temp = new double[4];

                        for (inch = 0; inch < inchannels - inchannels % 4; inch += 4) {
                            for (int i = 0; i < 4; i++) {
                                temp[i] = x[inch + i, ix, iy, th];
                            }

                            for (int outch = 0; outch < outchannels; outch++) {
                                double yv = y[outch, ix, iy, th];

                                for (int i = 0; i < 4; i++) {
                                    temp[i] += yv * w[inch + i, outch, 0, 0];
                                }
                            }

                            for (int i = 0; i < 4; i++) {
                                x[inch + i, ix, iy, th] = temp[i];
                            }
                        }

                        if (inchannels % 4 != 0) {
                            int sets = inchannels % 4;

                            for (int i = 0; i < sets; i++) {
                                temp[i] = x[inch + i, ix, iy, th];
                            }

                            for (int outch = 0; outch < outchannels; outch++) {
                                double yv = y[outch, ix, iy, th];

                                for (int i = 0; i < sets; i++) {
                                    temp[i] += yv * w[inch + i, outch, 0, 0];
                                }
                            }

                            for (int i = 0; i < sets; i++) {
                                x[inch + i, ix, iy, th] = temp[i];
                            }
                        }
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, inwidth = 13, inheight = 17, batch = 2;

            float[] yval = (new float[inwidth * inheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map2D y = new(outchannels, inwidth, inheight, batch, yval);
            Filter2D w = new(inchannels, outchannels, 1, 1, wval);

            Map2D x = Reference(y, w);

            float[] x_expect = {
                1.485000e-03f,  1.430000e-03f,  1.375000e-03f,  1.320000e-03f,  1.265000e-03f,  1.210000e-03f,  1.155000e-03f,
                6.446000e-03f,  6.270000e-03f,  6.094000e-03f,  5.918000e-03f,  5.742000e-03f,  5.566000e-03f,  5.390000e-03f,
                1.140700e-02f,  1.111000e-02f,  1.081300e-02f,  1.051600e-02f,  1.021900e-02f,  9.922000e-03f,  9.625000e-03f,
                1.636800e-02f,  1.595000e-02f,  1.553200e-02f,  1.511400e-02f,  1.469600e-02f,  1.427800e-02f,  1.386000e-02f,
                2.132900e-02f,  2.079000e-02f,  2.025100e-02f,  1.971200e-02f,  1.917300e-02f,  1.863400e-02f,  1.809500e-02f,
                2.629000e-02f,  2.563000e-02f,  2.497000e-02f,  2.431000e-02f,  2.365000e-02f,  2.299000e-02f,  2.233000e-02f,
                3.125100e-02f,  3.047000e-02f,  2.968900e-02f,  2.890800e-02f,  2.812700e-02f,  2.734600e-02f,  2.656500e-02f,
                3.621200e-02f,  3.531000e-02f,  3.440800e-02f,  3.350600e-02f,  3.260400e-02f,  3.170200e-02f,  3.080000e-02f,
                4.117300e-02f,  4.015000e-02f,  3.912700e-02f,  3.810400e-02f,  3.708100e-02f,  3.605800e-02f,  3.503500e-02f,
                4.613400e-02f,  4.499000e-02f,  4.384600e-02f,  4.270200e-02f,  4.155800e-02f,  4.041400e-02f,  3.927000e-02f,
                5.109500e-02f,  4.983000e-02f,  4.856500e-02f,  4.730000e-02f,  4.603500e-02f,  4.477000e-02f,  4.350500e-02f,
                5.605600e-02f,  5.467000e-02f,  5.328400e-02f,  5.189800e-02f,  5.051200e-02f,  4.912600e-02f,  4.774000e-02f,
                6.101700e-02f,  5.951000e-02f,  5.800300e-02f,  5.649600e-02f,  5.498900e-02f,  5.348200e-02f,  5.197500e-02f,
                6.597800e-02f,  6.435000e-02f,  6.272200e-02f,  6.109400e-02f,  5.946600e-02f,  5.783800e-02f,  5.621000e-02f,
                7.093900e-02f,  6.919000e-02f,  6.744100e-02f,  6.569200e-02f,  6.394300e-02f,  6.219400e-02f,  6.044500e-02f,
                7.590000e-02f,  7.403000e-02f,  7.216000e-02f,  7.029000e-02f,  6.842000e-02f,  6.655000e-02f,  6.468000e-02f,
                8.086100e-02f,  7.887000e-02f,  7.687900e-02f,  7.488800e-02f,  7.289700e-02f,  7.090600e-02f,  6.891500e-02f,
                8.582200e-02f,  8.371000e-02f,  8.159800e-02f,  7.948600e-02f,  7.737400e-02f,  7.526200e-02f,  7.315000e-02f,
                9.078300e-02f,  8.855000e-02f,  8.631700e-02f,  8.408400e-02f,  8.185100e-02f,  7.961800e-02f,  7.738500e-02f,
                9.574400e-02f,  9.339000e-02f,  9.103600e-02f,  8.868200e-02f,  8.632800e-02f,  8.397400e-02f,  8.162000e-02f,
                1.007050e-01f,  9.823000e-02f,  9.575500e-02f,  9.328000e-02f,  9.080500e-02f,  8.833000e-02f,  8.585500e-02f,
                1.056660e-01f,  1.030700e-01f,  1.004740e-01f,  9.787800e-02f,  9.528200e-02f,  9.268600e-02f,  9.009000e-02f,
                1.106270e-01f,  1.079100e-01f,  1.051930e-01f,  1.024760e-01f,  9.975900e-02f,  9.704200e-02f,  9.432500e-02f,
                1.155880e-01f,  1.127500e-01f,  1.099120e-01f,  1.070740e-01f,  1.042360e-01f,  1.013980e-01f,  9.856000e-02f,
                1.205490e-01f,  1.175900e-01f,  1.146310e-01f,  1.116720e-01f,  1.087130e-01f,  1.057540e-01f,  1.027950e-01f,
                1.255100e-01f,  1.224300e-01f,  1.193500e-01f,  1.162700e-01f,  1.131900e-01f,  1.101100e-01f,  1.070300e-01f,
                1.304710e-01f,  1.272700e-01f,  1.240690e-01f,  1.208680e-01f,  1.176670e-01f,  1.144660e-01f,  1.112650e-01f,
                1.354320e-01f,  1.321100e-01f,  1.287880e-01f,  1.254660e-01f,  1.221440e-01f,  1.188220e-01f,  1.155000e-01f,
                1.403930e-01f,  1.369500e-01f,  1.335070e-01f,  1.300640e-01f,  1.266210e-01f,  1.231780e-01f,  1.197350e-01f,
                1.453540e-01f,  1.417900e-01f,  1.382260e-01f,  1.346620e-01f,  1.310980e-01f,  1.275340e-01f,  1.239700e-01f,
                1.503150e-01f,  1.466300e-01f,  1.429450e-01f,  1.392600e-01f,  1.355750e-01f,  1.318900e-01f,  1.282050e-01f,
                1.552760e-01f,  1.514700e-01f,  1.476640e-01f,  1.438580e-01f,  1.400520e-01f,  1.362460e-01f,  1.324400e-01f,
                1.602370e-01f,  1.563100e-01f,  1.523830e-01f,  1.484560e-01f,  1.445290e-01f,  1.406020e-01f,  1.366750e-01f,
                1.651980e-01f,  1.611500e-01f,  1.571020e-01f,  1.530540e-01f,  1.490060e-01f,  1.449580e-01f,  1.409100e-01f,
                1.701590e-01f,  1.659900e-01f,  1.618210e-01f,  1.576520e-01f,  1.534830e-01f,  1.493140e-01f,  1.451450e-01f,
                1.751200e-01f,  1.708300e-01f,  1.665400e-01f,  1.622500e-01f,  1.579600e-01f,  1.536700e-01f,  1.493800e-01f,
                1.800810e-01f,  1.756700e-01f,  1.712590e-01f,  1.668480e-01f,  1.624370e-01f,  1.580260e-01f,  1.536150e-01f,
                1.850420e-01f,  1.805100e-01f,  1.759780e-01f,  1.714460e-01f,  1.669140e-01f,  1.623820e-01f,  1.578500e-01f,
                1.900030e-01f,  1.853500e-01f,  1.806970e-01f,  1.760440e-01f,  1.713910e-01f,  1.667380e-01f,  1.620850e-01f,
                1.949640e-01f,  1.901900e-01f,  1.854160e-01f,  1.806420e-01f,  1.758680e-01f,  1.710940e-01f,  1.663200e-01f,
                1.999250e-01f,  1.950300e-01f,  1.901350e-01f,  1.852400e-01f,  1.803450e-01f,  1.754500e-01f,  1.705550e-01f,
                2.048860e-01f,  1.998700e-01f,  1.948540e-01f,  1.898380e-01f,  1.848220e-01f,  1.798060e-01f,  1.747900e-01f,
                2.098470e-01f,  2.047100e-01f,  1.995730e-01f,  1.944360e-01f,  1.892990e-01f,  1.841620e-01f,  1.790250e-01f,
                2.148080e-01f,  2.095500e-01f,  2.042920e-01f,  1.990340e-01f,  1.937760e-01f,  1.885180e-01f,  1.832600e-01f,
                2.197690e-01f,  2.143900e-01f,  2.090110e-01f,  2.036320e-01f,  1.982530e-01f,  1.928740e-01f,  1.874950e-01f,
                2.247300e-01f,  2.192300e-01f,  2.137300e-01f,  2.082300e-01f,  2.027300e-01f,  1.972300e-01f,  1.917300e-01f,
                2.296910e-01f,  2.240700e-01f,  2.184490e-01f,  2.128280e-01f,  2.072070e-01f,  2.015860e-01f,  1.959650e-01f,
                2.346520e-01f,  2.289100e-01f,  2.231680e-01f,  2.174260e-01f,  2.116840e-01f,  2.059420e-01f,  2.002000e-01f,
                2.396130e-01f,  2.337500e-01f,  2.278870e-01f,  2.220240e-01f,  2.161610e-01f,  2.102980e-01f,  2.044350e-01f,
                2.445740e-01f,  2.385900e-01f,  2.326060e-01f,  2.266220e-01f,  2.206380e-01f,  2.146540e-01f,  2.086700e-01f,
                2.495350e-01f,  2.434300e-01f,  2.373250e-01f,  2.312200e-01f,  2.251150e-01f,  2.190100e-01f,  2.129050e-01f,
                2.544960e-01f,  2.482700e-01f,  2.420440e-01f,  2.358180e-01f,  2.295920e-01f,  2.233660e-01f,  2.171400e-01f,
                2.594570e-01f,  2.531100e-01f,  2.467630e-01f,  2.404160e-01f,  2.340690e-01f,  2.277220e-01f,  2.213750e-01f,
                2.644180e-01f,  2.579500e-01f,  2.514820e-01f,  2.450140e-01f,  2.385460e-01f,  2.320780e-01f,  2.256100e-01f,
                2.693790e-01f,  2.627900e-01f,  2.562010e-01f,  2.496120e-01f,  2.430230e-01f,  2.364340e-01f,  2.298450e-01f,
                2.743400e-01f,  2.676300e-01f,  2.609200e-01f,  2.542100e-01f,  2.475000e-01f,  2.407900e-01f,  2.340800e-01f,
                2.793010e-01f,  2.724700e-01f,  2.656390e-01f,  2.588080e-01f,  2.519770e-01f,  2.451460e-01f,  2.383150e-01f,
                2.842620e-01f,  2.773100e-01f,  2.703580e-01f,  2.634060e-01f,  2.564540e-01f,  2.495020e-01f,  2.425500e-01f,
                2.892230e-01f,  2.821500e-01f,  2.750770e-01f,  2.680040e-01f,  2.609310e-01f,  2.538580e-01f,  2.467850e-01f,
                2.941840e-01f,  2.869900e-01f,  2.797960e-01f,  2.726020e-01f,  2.654080e-01f,  2.582140e-01f,  2.510200e-01f,
                2.991450e-01f,  2.918300e-01f,  2.845150e-01f,  2.772000e-01f,  2.698850e-01f,  2.625700e-01f,  2.552550e-01f,
                3.041060e-01f,  2.966700e-01f,  2.892340e-01f,  2.817980e-01f,  2.743620e-01f,  2.669260e-01f,  2.594900e-01f,
                3.090670e-01f,  3.015100e-01f,  2.939530e-01f,  2.863960e-01f,  2.788390e-01f,  2.712820e-01f,  2.637250e-01f,
                3.140280e-01f,  3.063500e-01f,  2.986720e-01f,  2.909940e-01f,  2.833160e-01f,  2.756380e-01f,  2.679600e-01f,
                3.189890e-01f,  3.111900e-01f,  3.033910e-01f,  2.955920e-01f,  2.877930e-01f,  2.799940e-01f,  2.721950e-01f,
                3.239500e-01f,  3.160300e-01f,  3.081100e-01f,  3.001900e-01f,  2.922700e-01f,  2.843500e-01f,  2.764300e-01f,
                3.289110e-01f,  3.208700e-01f,  3.128290e-01f,  3.047880e-01f,  2.967470e-01f,  2.887060e-01f,  2.806650e-01f,
                3.338720e-01f,  3.257100e-01f,  3.175480e-01f,  3.093860e-01f,  3.012240e-01f,  2.930620e-01f,  2.849000e-01f,
                3.388330e-01f,  3.305500e-01f,  3.222670e-01f,  3.139840e-01f,  3.057010e-01f,  2.974180e-01f,  2.891350e-01f,
                3.437940e-01f,  3.353900e-01f,  3.269860e-01f,  3.185820e-01f,  3.101780e-01f,  3.017740e-01f,  2.933700e-01f,
                3.487550e-01f,  3.402300e-01f,  3.317050e-01f,  3.231800e-01f,  3.146550e-01f,  3.061300e-01f,  2.976050e-01f,
                3.537160e-01f,  3.450700e-01f,  3.364240e-01f,  3.277780e-01f,  3.191320e-01f,  3.104860e-01f,  3.018400e-01f,
                3.586770e-01f,  3.499100e-01f,  3.411430e-01f,  3.323760e-01f,  3.236090e-01f,  3.148420e-01f,  3.060750e-01f,
                3.636380e-01f,  3.547500e-01f,  3.458620e-01f,  3.369740e-01f,  3.280860e-01f,  3.191980e-01f,  3.103100e-01f,
                3.685990e-01f,  3.595900e-01f,  3.505810e-01f,  3.415720e-01f,  3.325630e-01f,  3.235540e-01f,  3.145450e-01f,
                3.735600e-01f,  3.644300e-01f,  3.553000e-01f,  3.461700e-01f,  3.370400e-01f,  3.279100e-01f,  3.187800e-01f,
                3.785210e-01f,  3.692700e-01f,  3.600190e-01f,  3.507680e-01f,  3.415170e-01f,  3.322660e-01f,  3.230150e-01f,
                3.834820e-01f,  3.741100e-01f,  3.647380e-01f,  3.553660e-01f,  3.459940e-01f,  3.366220e-01f,  3.272500e-01f,
                3.884430e-01f,  3.789500e-01f,  3.694570e-01f,  3.599640e-01f,  3.504710e-01f,  3.409780e-01f,  3.314850e-01f,
                3.934040e-01f,  3.837900e-01f,  3.741760e-01f,  3.645620e-01f,  3.549480e-01f,  3.453340e-01f,  3.357200e-01f,
                3.983650e-01f,  3.886300e-01f,  3.788950e-01f,  3.691600e-01f,  3.594250e-01f,  3.496900e-01f,  3.399550e-01f,
                4.033260e-01f,  3.934700e-01f,  3.836140e-01f,  3.737580e-01f,  3.639020e-01f,  3.540460e-01f,  3.441900e-01f,
                4.082870e-01f,  3.983100e-01f,  3.883330e-01f,  3.783560e-01f,  3.683790e-01f,  3.584020e-01f,  3.484250e-01f,
                4.132480e-01f,  4.031500e-01f,  3.930520e-01f,  3.829540e-01f,  3.728560e-01f,  3.627580e-01f,  3.526600e-01f,
                4.182090e-01f,  4.079900e-01f,  3.977710e-01f,  3.875520e-01f,  3.773330e-01f,  3.671140e-01f,  3.568950e-01f,
                4.231700e-01f,  4.128300e-01f,  4.024900e-01f,  3.921500e-01f,  3.818100e-01f,  3.714700e-01f,  3.611300e-01f,
                4.281310e-01f,  4.176700e-01f,  4.072090e-01f,  3.967480e-01f,  3.862870e-01f,  3.758260e-01f,  3.653650e-01f,
                4.330920e-01f,  4.225100e-01f,  4.119280e-01f,  4.013460e-01f,  3.907640e-01f,  3.801820e-01f,  3.696000e-01f,
                4.380530e-01f,  4.273500e-01f,  4.166470e-01f,  4.059440e-01f,  3.952410e-01f,  3.845380e-01f,  3.738350e-01f,
                4.430140e-01f,  4.321900e-01f,  4.213660e-01f,  4.105420e-01f,  3.997180e-01f,  3.888940e-01f,  3.780700e-01f,
                4.479750e-01f,  4.370300e-01f,  4.260850e-01f,  4.151400e-01f,  4.041950e-01f,  3.932500e-01f,  3.823050e-01f,
                4.529360e-01f,  4.418700e-01f,  4.308040e-01f,  4.197380e-01f,  4.086720e-01f,  3.976060e-01f,  3.865400e-01f,
                4.578970e-01f,  4.467100e-01f,  4.355230e-01f,  4.243360e-01f,  4.131490e-01f,  4.019620e-01f,  3.907750e-01f,
                4.628580e-01f,  4.515500e-01f,  4.402420e-01f,  4.289340e-01f,  4.176260e-01f,  4.063180e-01f,  3.950100e-01f,
                4.678190e-01f,  4.563900e-01f,  4.449610e-01f,  4.335320e-01f,  4.221030e-01f,  4.106740e-01f,  3.992450e-01f,
                4.727800e-01f,  4.612300e-01f,  4.496800e-01f,  4.381300e-01f,  4.265800e-01f,  4.150300e-01f,  4.034800e-01f,
                4.777410e-01f,  4.660700e-01f,  4.543990e-01f,  4.427280e-01f,  4.310570e-01f,  4.193860e-01f,  4.077150e-01f,
                4.827020e-01f,  4.709100e-01f,  4.591180e-01f,  4.473260e-01f,  4.355340e-01f,  4.237420e-01f,  4.119500e-01f,
                4.876630e-01f,  4.757500e-01f,  4.638370e-01f,  4.519240e-01f,  4.400110e-01f,  4.280980e-01f,  4.161850e-01f,
                4.926240e-01f,  4.805900e-01f,  4.685560e-01f,  4.565220e-01f,  4.444880e-01f,  4.324540e-01f,  4.204200e-01f,
                4.975850e-01f,  4.854300e-01f,  4.732750e-01f,  4.611200e-01f,  4.489650e-01f,  4.368100e-01f,  4.246550e-01f,
                5.025460e-01f,  4.902700e-01f,  4.779940e-01f,  4.657180e-01f,  4.534420e-01f,  4.411660e-01f,  4.288900e-01f,
                5.075070e-01f,  4.951100e-01f,  4.827130e-01f,  4.703160e-01f,  4.579190e-01f,  4.455220e-01f,  4.331250e-01f,
                5.124680e-01f,  4.999500e-01f,  4.874320e-01f,  4.749140e-01f,  4.623960e-01f,  4.498780e-01f,  4.373600e-01f,
                5.174290e-01f,  5.047900e-01f,  4.921510e-01f,  4.795120e-01f,  4.668730e-01f,  4.542340e-01f,  4.415950e-01f,
                5.223900e-01f,  5.096300e-01f,  4.968700e-01f,  4.841100e-01f,  4.713500e-01f,  4.585900e-01f,  4.458300e-01f,
                5.273510e-01f,  5.144700e-01f,  5.015890e-01f,  4.887080e-01f,  4.758270e-01f,  4.629460e-01f,  4.500650e-01f,
                5.323120e-01f,  5.193100e-01f,  5.063080e-01f,  4.933060e-01f,  4.803040e-01f,  4.673020e-01f,  4.543000e-01f,
                5.372730e-01f,  5.241500e-01f,  5.110270e-01f,  4.979040e-01f,  4.847810e-01f,  4.716580e-01f,  4.585350e-01f,
                5.422340e-01f,  5.289900e-01f,  5.157460e-01f,  5.025020e-01f,  4.892580e-01f,  4.760140e-01f,  4.627700e-01f,
                5.471950e-01f,  5.338300e-01f,  5.204650e-01f,  5.071000e-01f,  4.937350e-01f,  4.803700e-01f,  4.670050e-01f,
                5.521560e-01f,  5.386700e-01f,  5.251840e-01f,  5.116980e-01f,  4.982120e-01f,  4.847260e-01f,  4.712400e-01f,
                5.571170e-01f,  5.435100e-01f,  5.299030e-01f,  5.162960e-01f,  5.026890e-01f,  4.890820e-01f,  4.754750e-01f,
                5.620780e-01f,  5.483500e-01f,  5.346220e-01f,  5.208940e-01f,  5.071660e-01f,  4.934380e-01f,  4.797100e-01f,
                5.670390e-01f,  5.531900e-01f,  5.393410e-01f,  5.254920e-01f,  5.116430e-01f,  4.977940e-01f,  4.839450e-01f,
                5.720000e-01f,  5.580300e-01f,  5.440600e-01f,  5.300900e-01f,  5.161200e-01f,  5.021500e-01f,  4.881800e-01f,
                5.769610e-01f,  5.628700e-01f,  5.487790e-01f,  5.346880e-01f,  5.205970e-01f,  5.065060e-01f,  4.924150e-01f,
                5.819220e-01f,  5.677100e-01f,  5.534980e-01f,  5.392860e-01f,  5.250740e-01f,  5.108620e-01f,  4.966500e-01f,
                5.868830e-01f,  5.725500e-01f,  5.582170e-01f,  5.438840e-01f,  5.295510e-01f,  5.152180e-01f,  5.008850e-01f,
                5.918440e-01f,  5.773900e-01f,  5.629360e-01f,  5.484820e-01f,  5.340280e-01f,  5.195740e-01f,  5.051200e-01f,
                5.968050e-01f,  5.822300e-01f,  5.676550e-01f,  5.530800e-01f,  5.385050e-01f,  5.239300e-01f,  5.093550e-01f,
                6.017660e-01f,  5.870700e-01f,  5.723740e-01f,  5.576780e-01f,  5.429820e-01f,  5.282860e-01f,  5.135900e-01f,
                6.067270e-01f,  5.919100e-01f,  5.770930e-01f,  5.622760e-01f,  5.474590e-01f,  5.326420e-01f,  5.178250e-01f,
                6.116880e-01f,  5.967500e-01f,  5.818120e-01f,  5.668740e-01f,  5.519360e-01f,  5.369980e-01f,  5.220600e-01f,
                6.166490e-01f,  6.015900e-01f,  5.865310e-01f,  5.714720e-01f,  5.564130e-01f,  5.413540e-01f,  5.262950e-01f,
                6.216100e-01f,  6.064300e-01f,  5.912500e-01f,  5.760700e-01f,  5.608900e-01f,  5.457100e-01f,  5.305300e-01f,
                6.265710e-01f,  6.112700e-01f,  5.959690e-01f,  5.806680e-01f,  5.653670e-01f,  5.500660e-01f,  5.347650e-01f,
                6.315320e-01f,  6.161100e-01f,  6.006880e-01f,  5.852660e-01f,  5.698440e-01f,  5.544220e-01f,  5.390000e-01f,
                6.364930e-01f,  6.209500e-01f,  6.054070e-01f,  5.898640e-01f,  5.743210e-01f,  5.587780e-01f,  5.432350e-01f,
                6.414540e-01f,  6.257900e-01f,  6.101260e-01f,  5.944620e-01f,  5.787980e-01f,  5.631340e-01f,  5.474700e-01f,
                6.464150e-01f,  6.306300e-01f,  6.148450e-01f,  5.990600e-01f,  5.832750e-01f,  5.674900e-01f,  5.517050e-01f,
                6.513760e-01f,  6.354700e-01f,  6.195640e-01f,  6.036580e-01f,  5.877520e-01f,  5.718460e-01f,  5.559400e-01f,
                6.563370e-01f,  6.403100e-01f,  6.242830e-01f,  6.082560e-01f,  5.922290e-01f,  5.762020e-01f,  5.601750e-01f,
                6.612980e-01f,  6.451500e-01f,  6.290020e-01f,  6.128540e-01f,  5.967060e-01f,  5.805580e-01f,  5.644100e-01f,
                6.662590e-01f,  6.499900e-01f,  6.337210e-01f,  6.174520e-01f,  6.011830e-01f,  5.849140e-01f,  5.686450e-01f,
                6.712200e-01f,  6.548300e-01f,  6.384400e-01f,  6.220500e-01f,  6.056600e-01f,  5.892700e-01f,  5.728800e-01f,
                6.761810e-01f,  6.596700e-01f,  6.431590e-01f,  6.266480e-01f,  6.101370e-01f,  5.936260e-01f,  5.771150e-01f,
                6.811420e-01f,  6.645100e-01f,  6.478780e-01f,  6.312460e-01f,  6.146140e-01f,  5.979820e-01f,  5.813500e-01f,
                6.861030e-01f,  6.693500e-01f,  6.525970e-01f,  6.358440e-01f,  6.190910e-01f,  6.023380e-01f,  5.855850e-01f,
                6.910640e-01f,  6.741900e-01f,  6.573160e-01f,  6.404420e-01f,  6.235680e-01f,  6.066940e-01f,  5.898200e-01f,
                6.960250e-01f,  6.790300e-01f,  6.620350e-01f,  6.450400e-01f,  6.280450e-01f,  6.110500e-01f,  5.940550e-01f,
                7.009860e-01f,  6.838700e-01f,  6.667540e-01f,  6.496380e-01f,  6.325220e-01f,  6.154060e-01f,  5.982900e-01f,
                7.059470e-01f,  6.887100e-01f,  6.714730e-01f,  6.542360e-01f,  6.369990e-01f,  6.197620e-01f,  6.025250e-01f,
                7.109080e-01f,  6.935500e-01f,  6.761920e-01f,  6.588340e-01f,  6.414760e-01f,  6.241180e-01f,  6.067600e-01f,
                7.158690e-01f,  6.983900e-01f,  6.809110e-01f,  6.634320e-01f,  6.459530e-01f,  6.284740e-01f,  6.109950e-01f,
                7.208300e-01f,  7.032300e-01f,  6.856300e-01f,  6.680300e-01f,  6.504300e-01f,  6.328300e-01f,  6.152300e-01f,
                7.257910e-01f,  7.080700e-01f,  6.903490e-01f,  6.726280e-01f,  6.549070e-01f,  6.371860e-01f,  6.194650e-01f,
                7.307520e-01f,  7.129100e-01f,  6.950680e-01f,  6.772260e-01f,  6.593840e-01f,  6.415420e-01f,  6.237000e-01f,
                7.357130e-01f,  7.177500e-01f,  6.997870e-01f,  6.818240e-01f,  6.638610e-01f,  6.458980e-01f,  6.279350e-01f,
                7.406740e-01f,  7.225900e-01f,  7.045060e-01f,  6.864220e-01f,  6.683380e-01f,  6.502540e-01f,  6.321700e-01f,
                7.456350e-01f,  7.274300e-01f,  7.092250e-01f,  6.910200e-01f,  6.728150e-01f,  6.546100e-01f,  6.364050e-01f,
                7.505960e-01f,  7.322700e-01f,  7.139440e-01f,  6.956180e-01f,  6.772920e-01f,  6.589660e-01f,  6.406400e-01f,
                7.555570e-01f,  7.371100e-01f,  7.186630e-01f,  7.002160e-01f,  6.817690e-01f,  6.633220e-01f,  6.448750e-01f,
                7.605180e-01f,  7.419500e-01f,  7.233820e-01f,  7.048140e-01f,  6.862460e-01f,  6.676780e-01f,  6.491100e-01f,
                7.654790e-01f,  7.467900e-01f,  7.281010e-01f,  7.094120e-01f,  6.907230e-01f,  6.720340e-01f,  6.533450e-01f,
                7.704400e-01f,  7.516300e-01f,  7.328200e-01f,  7.140100e-01f,  6.952000e-01f,  6.763900e-01f,  6.575800e-01f,
                7.754010e-01f,  7.564700e-01f,  7.375390e-01f,  7.186080e-01f,  6.996770e-01f,  6.807460e-01f,  6.618150e-01f,
                7.803620e-01f,  7.613100e-01f,  7.422580e-01f,  7.232060e-01f,  7.041540e-01f,  6.851020e-01f,  6.660500e-01f,
                7.853230e-01f,  7.661500e-01f,  7.469770e-01f,  7.278040e-01f,  7.086310e-01f,  6.894580e-01f,  6.702850e-01f,
                7.902840e-01f,  7.709900e-01f,  7.516960e-01f,  7.324020e-01f,  7.131080e-01f,  6.938140e-01f,  6.745200e-01f,
                7.952450e-01f,  7.758300e-01f,  7.564150e-01f,  7.370000e-01f,  7.175850e-01f,  6.981700e-01f,  6.787550e-01f,
                8.002060e-01f,  7.806700e-01f,  7.611340e-01f,  7.415980e-01f,  7.220620e-01f,  7.025260e-01f,  6.829900e-01f,
                8.051670e-01f,  7.855100e-01f,  7.658530e-01f,  7.461960e-01f,  7.265390e-01f,  7.068820e-01f,  6.872250e-01f,
                8.101280e-01f,  7.903500e-01f,  7.705720e-01f,  7.507940e-01f,  7.310160e-01f,  7.112380e-01f,  6.914600e-01f,
                8.150890e-01f,  7.951900e-01f,  7.752910e-01f,  7.553920e-01f,  7.354930e-01f,  7.155940e-01f,  6.956950e-01f,
                8.200500e-01f,  8.000300e-01f,  7.800100e-01f,  7.599900e-01f,  7.399700e-01f,  7.199500e-01f,  6.999300e-01f,
                8.250110e-01f,  8.048700e-01f,  7.847290e-01f,  7.645880e-01f,  7.444470e-01f,  7.243060e-01f,  7.041650e-01f,
                8.299720e-01f,  8.097100e-01f,  7.894480e-01f,  7.691860e-01f,  7.489240e-01f,  7.286620e-01f,  7.084000e-01f,
                8.349330e-01f,  8.145500e-01f,  7.941670e-01f,  7.737840e-01f,  7.534010e-01f,  7.330180e-01f,  7.126350e-01f,
                8.398940e-01f,  8.193900e-01f,  7.988860e-01f,  7.783820e-01f,  7.578780e-01f,  7.373740e-01f,  7.168700e-01f,
                8.448550e-01f,  8.242300e-01f,  8.036050e-01f,  7.829800e-01f,  7.623550e-01f,  7.417300e-01f,  7.211050e-01f,
                8.498160e-01f,  8.290700e-01f,  8.083240e-01f,  7.875780e-01f,  7.668320e-01f,  7.460860e-01f,  7.253400e-01f,
                8.547770e-01f,  8.339100e-01f,  8.130430e-01f,  7.921760e-01f,  7.713090e-01f,  7.504420e-01f,  7.295750e-01f,
                8.597380e-01f,  8.387500e-01f,  8.177620e-01f,  7.967740e-01f,  7.757860e-01f,  7.547980e-01f,  7.338100e-01f,
                8.646990e-01f,  8.435900e-01f,  8.224810e-01f,  8.013720e-01f,  7.802630e-01f,  7.591540e-01f,  7.380450e-01f,
                8.696600e-01f,  8.484300e-01f,  8.272000e-01f,  8.059700e-01f,  7.847400e-01f,  7.635100e-01f,  7.422800e-01f,
                8.746210e-01f,  8.532700e-01f,  8.319190e-01f,  8.105680e-01f,  7.892170e-01f,  7.678660e-01f,  7.465150e-01f,
                8.795820e-01f,  8.581100e-01f,  8.366380e-01f,  8.151660e-01f,  7.936940e-01f,  7.722220e-01f,  7.507500e-01f,
                8.845430e-01f,  8.629500e-01f,  8.413570e-01f,  8.197640e-01f,  7.981710e-01f,  7.765780e-01f,  7.549850e-01f,
                8.895040e-01f,  8.677900e-01f,  8.460760e-01f,  8.243620e-01f,  8.026480e-01f,  7.809340e-01f,  7.592200e-01f,
                8.944650e-01f,  8.726300e-01f,  8.507950e-01f,  8.289600e-01f,  8.071250e-01f,  7.852900e-01f,  7.634550e-01f,
                8.994260e-01f,  8.774700e-01f,  8.555140e-01f,  8.335580e-01f,  8.116020e-01f,  7.896460e-01f,  7.676900e-01f,
                9.043870e-01f,  8.823100e-01f,  8.602330e-01f,  8.381560e-01f,  8.160790e-01f,  7.940020e-01f,  7.719250e-01f,
                9.093480e-01f,  8.871500e-01f,  8.649520e-01f,  8.427540e-01f,  8.205560e-01f,  7.983580e-01f,  7.761600e-01f,
                9.143090e-01f,  8.919900e-01f,  8.696710e-01f,  8.473520e-01f,  8.250330e-01f,  8.027140e-01f,  7.803950e-01f,
                9.192700e-01f,  8.968300e-01f,  8.743900e-01f,  8.519500e-01f,  8.295100e-01f,  8.070700e-01f,  7.846300e-01f,
                9.242310e-01f,  9.016700e-01f,  8.791090e-01f,  8.565480e-01f,  8.339870e-01f,  8.114260e-01f,  7.888650e-01f,
                9.291920e-01f,  9.065100e-01f,  8.838280e-01f,  8.611460e-01f,  8.384640e-01f,  8.157820e-01f,  7.931000e-01f,
                9.341530e-01f,  9.113500e-01f,  8.885470e-01f,  8.657440e-01f,  8.429410e-01f,  8.201380e-01f,  7.973350e-01f,
                9.391140e-01f,  9.161900e-01f,  8.932660e-01f,  8.703420e-01f,  8.474180e-01f,  8.244940e-01f,  8.015700e-01f,
                9.440750e-01f,  9.210300e-01f,  8.979850e-01f,  8.749400e-01f,  8.518950e-01f,  8.288500e-01f,  8.058050e-01f,
                9.490360e-01f,  9.258700e-01f,  9.027040e-01f,  8.795380e-01f,  8.563720e-01f,  8.332060e-01f,  8.100400e-01f,
                9.539970e-01f,  9.307100e-01f,  9.074230e-01f,  8.841360e-01f,  8.608490e-01f,  8.375620e-01f,  8.142750e-01f,
                9.589580e-01f,  9.355500e-01f,  9.121420e-01f,  8.887340e-01f,  8.653260e-01f,  8.419180e-01f,  8.185100e-01f,
                9.639190e-01f,  9.403900e-01f,  9.168610e-01f,  8.933320e-01f,  8.698030e-01f,  8.462740e-01f,  8.227450e-01f,
                9.688800e-01f,  9.452300e-01f,  9.215800e-01f,  8.979300e-01f,  8.742800e-01f,  8.506300e-01f,  8.269800e-01f,
                9.738410e-01f,  9.500700e-01f,  9.262990e-01f,  9.025280e-01f,  8.787570e-01f,  8.549860e-01f,  8.312150e-01f,
                9.788020e-01f,  9.549100e-01f,  9.310180e-01f,  9.071260e-01f,  8.832340e-01f,  8.593420e-01f,  8.354500e-01f,
                9.837630e-01f,  9.597500e-01f,  9.357370e-01f,  9.117240e-01f,  8.877110e-01f,  8.636980e-01f,  8.396850e-01f,
                9.887240e-01f,  9.645900e-01f,  9.404560e-01f,  9.163220e-01f,  8.921880e-01f,  8.680540e-01f,  8.439200e-01f,
                9.936850e-01f,  9.694300e-01f,  9.451750e-01f,  9.209200e-01f,  8.966650e-01f,  8.724100e-01f,  8.481550e-01f,
                9.986460e-01f,  9.742700e-01f,  9.498940e-01f,  9.255180e-01f,  9.011420e-01f,  8.767660e-01f,  8.523900e-01f,
                1.003607e+00f,  9.791100e-01f,  9.546130e-01f,  9.301160e-01f,  9.056190e-01f,  8.811220e-01f,  8.566250e-01f,
                1.008568e+00f,  9.839500e-01f,  9.593320e-01f,  9.347140e-01f,  9.100960e-01f,  8.854780e-01f,  8.608600e-01f,
                1.013529e+00f,  9.887900e-01f,  9.640510e-01f,  9.393120e-01f,  9.145730e-01f,  8.898340e-01f,  8.650950e-01f,
                1.018490e+00f,  9.936300e-01f,  9.687700e-01f,  9.439100e-01f,  9.190500e-01f,  8.941900e-01f,  8.693300e-01f,
                1.023451e+00f,  9.984700e-01f,  9.734890e-01f,  9.485080e-01f,  9.235270e-01f,  8.985460e-01f,  8.735650e-01f,
                1.028412e+00f,  1.003310e+00f,  9.782080e-01f,  9.531060e-01f,  9.280040e-01f,  9.029020e-01f,  8.778000e-01f,
                1.033373e+00f,  1.008150e+00f,  9.829270e-01f,  9.577040e-01f,  9.324810e-01f,  9.072580e-01f,  8.820350e-01f,
                1.038334e+00f,  1.012990e+00f,  9.876460e-01f,  9.623020e-01f,  9.369580e-01f,  9.116140e-01f,  8.862700e-01f,
                1.043295e+00f,  1.017830e+00f,  9.923650e-01f,  9.669000e-01f,  9.414350e-01f,  9.159700e-01f,  8.905050e-01f,
                1.048256e+00f,  1.022670e+00f,  9.970840e-01f,  9.714980e-01f,  9.459120e-01f,  9.203260e-01f,  8.947400e-01f,
                1.053217e+00f,  1.027510e+00f,  1.001803e+00f,  9.760960e-01f,  9.503890e-01f,  9.246820e-01f,  8.989750e-01f,
                1.058178e+00f,  1.032350e+00f,  1.006522e+00f,  9.806940e-01f,  9.548660e-01f,  9.290380e-01f,  9.032100e-01f,
                1.063139e+00f,  1.037190e+00f,  1.011241e+00f,  9.852920e-01f,  9.593430e-01f,  9.333940e-01f,  9.074450e-01f,
                1.068100e+00f,  1.042030e+00f,  1.015960e+00f,  9.898900e-01f,  9.638200e-01f,  9.377500e-01f,  9.116800e-01f,
                1.073061e+00f,  1.046870e+00f,  1.020679e+00f,  9.944880e-01f,  9.682970e-01f,  9.421060e-01f,  9.159150e-01f,
                1.078022e+00f,  1.051710e+00f,  1.025398e+00f,  9.990860e-01f,  9.727740e-01f,  9.464620e-01f,  9.201500e-01f,
                1.082983e+00f,  1.056550e+00f,  1.030117e+00f,  1.003684e+00f,  9.772510e-01f,  9.508180e-01f,  9.243850e-01f,
                1.087944e+00f,  1.061390e+00f,  1.034836e+00f,  1.008282e+00f,  9.817280e-01f,  9.551740e-01f,  9.286200e-01f,
                1.092905e+00f,  1.066230e+00f,  1.039555e+00f,  1.012880e+00f,  9.862050e-01f,  9.595300e-01f,  9.328550e-01f,
                1.097866e+00f,  1.071070e+00f,  1.044274e+00f,  1.017478e+00f,  9.906820e-01f,  9.638860e-01f,  9.370900e-01f,
                1.102827e+00f,  1.075910e+00f,  1.048993e+00f,  1.022076e+00f,  9.951590e-01f,  9.682420e-01f,  9.413250e-01f,
                1.107788e+00f,  1.080750e+00f,  1.053712e+00f,  1.026674e+00f,  9.996360e-01f,  9.725980e-01f,  9.455600e-01f,
                1.112749e+00f,  1.085590e+00f,  1.058431e+00f,  1.031272e+00f,  1.004113e+00f,  9.769540e-01f,  9.497950e-01f,
                1.117710e+00f,  1.090430e+00f,  1.063150e+00f,  1.035870e+00f,  1.008590e+00f,  9.813100e-01f,  9.540300e-01f,
                1.122671e+00f,  1.095270e+00f,  1.067869e+00f,  1.040468e+00f,  1.013067e+00f,  9.856660e-01f,  9.582650e-01f,
                1.127632e+00f,  1.100110e+00f,  1.072588e+00f,  1.045066e+00f,  1.017544e+00f,  9.900220e-01f,  9.625000e-01f,
                1.132593e+00f,  1.104950e+00f,  1.077307e+00f,  1.049664e+00f,  1.022021e+00f,  9.943780e-01f,  9.667350e-01f,
                1.137554e+00f,  1.109790e+00f,  1.082026e+00f,  1.054262e+00f,  1.026498e+00f,  9.987340e-01f,  9.709700e-01f,
                1.142515e+00f,  1.114630e+00f,  1.086745e+00f,  1.058860e+00f,  1.030975e+00f,  1.003090e+00f,  9.752050e-01f,
                1.147476e+00f,  1.119470e+00f,  1.091464e+00f,  1.063458e+00f,  1.035452e+00f,  1.007446e+00f,  9.794400e-01f,
                1.152437e+00f,  1.124310e+00f,  1.096183e+00f,  1.068056e+00f,  1.039929e+00f,  1.011802e+00f,  9.836750e-01f,
                1.157398e+00f,  1.129150e+00f,  1.100902e+00f,  1.072654e+00f,  1.044406e+00f,  1.016158e+00f,  9.879100e-01f,
                1.162359e+00f,  1.133990e+00f,  1.105621e+00f,  1.077252e+00f,  1.048883e+00f,  1.020514e+00f,  9.921450e-01f,
                1.167320e+00f,  1.138830e+00f,  1.110340e+00f,  1.081850e+00f,  1.053360e+00f,  1.024870e+00f,  9.963800e-01f,
                1.172281e+00f,  1.143670e+00f,  1.115059e+00f,  1.086448e+00f,  1.057837e+00f,  1.029226e+00f,  1.000615e+00f,
                1.177242e+00f,  1.148510e+00f,  1.119778e+00f,  1.091046e+00f,  1.062314e+00f,  1.033582e+00f,  1.004850e+00f,
                1.182203e+00f,  1.153350e+00f,  1.124497e+00f,  1.095644e+00f,  1.066791e+00f,  1.037938e+00f,  1.009085e+00f,
                1.187164e+00f,  1.158190e+00f,  1.129216e+00f,  1.100242e+00f,  1.071268e+00f,  1.042294e+00f,  1.013320e+00f,
                1.192125e+00f,  1.163030e+00f,  1.133935e+00f,  1.104840e+00f,  1.075745e+00f,  1.046650e+00f,  1.017555e+00f,
                1.197086e+00f,  1.167870e+00f,  1.138654e+00f,  1.109438e+00f,  1.080222e+00f,  1.051006e+00f,  1.021790e+00f,
                1.202047e+00f,  1.172710e+00f,  1.143373e+00f,  1.114036e+00f,  1.084699e+00f,  1.055362e+00f,  1.026025e+00f,
                1.207008e+00f,  1.177550e+00f,  1.148092e+00f,  1.118634e+00f,  1.089176e+00f,  1.059718e+00f,  1.030260e+00f,
                1.211969e+00f,  1.182390e+00f,  1.152811e+00f,  1.123232e+00f,  1.093653e+00f,  1.064074e+00f,  1.034495e+00f,
                1.216930e+00f,  1.187230e+00f,  1.157530e+00f,  1.127830e+00f,  1.098130e+00f,  1.068430e+00f,  1.038730e+00f,
                1.221891e+00f,  1.192070e+00f,  1.162249e+00f,  1.132428e+00f,  1.102607e+00f,  1.072786e+00f,  1.042965e+00f,
                1.226852e+00f,  1.196910e+00f,  1.166968e+00f,  1.137026e+00f,  1.107084e+00f,  1.077142e+00f,  1.047200e+00f,
                1.231813e+00f,  1.201750e+00f,  1.171687e+00f,  1.141624e+00f,  1.111561e+00f,  1.081498e+00f,  1.051435e+00f,
                1.236774e+00f,  1.206590e+00f,  1.176406e+00f,  1.146222e+00f,  1.116038e+00f,  1.085854e+00f,  1.055670e+00f,
                1.241735e+00f,  1.211430e+00f,  1.181125e+00f,  1.150820e+00f,  1.120515e+00f,  1.090210e+00f,  1.059905e+00f,
                1.246696e+00f,  1.216270e+00f,  1.185844e+00f,  1.155418e+00f,  1.124992e+00f,  1.094566e+00f,  1.064140e+00f,
                1.251657e+00f,  1.221110e+00f,  1.190563e+00f,  1.160016e+00f,  1.129469e+00f,  1.098922e+00f,  1.068375e+00f,
                1.256618e+00f,  1.225950e+00f,  1.195282e+00f,  1.164614e+00f,  1.133946e+00f,  1.103278e+00f,  1.072610e+00f,
                1.261579e+00f,  1.230790e+00f,  1.200001e+00f,  1.169212e+00f,  1.138423e+00f,  1.107634e+00f,  1.076845e+00f,
                1.266540e+00f,  1.235630e+00f,  1.204720e+00f,  1.173810e+00f,  1.142900e+00f,  1.111990e+00f,  1.081080e+00f,
                1.271501e+00f,  1.240470e+00f,  1.209439e+00f,  1.178408e+00f,  1.147377e+00f,  1.116346e+00f,  1.085315e+00f,
                1.276462e+00f,  1.245310e+00f,  1.214158e+00f,  1.183006e+00f,  1.151854e+00f,  1.120702e+00f,  1.089550e+00f,
                1.281423e+00f,  1.250150e+00f,  1.218877e+00f,  1.187604e+00f,  1.156331e+00f,  1.125058e+00f,  1.093785e+00f,
                1.286384e+00f,  1.254990e+00f,  1.223596e+00f,  1.192202e+00f,  1.160808e+00f,  1.129414e+00f,  1.098020e+00f,
                1.291345e+00f,  1.259830e+00f,  1.228315e+00f,  1.196800e+00f,  1.165285e+00f,  1.133770e+00f,  1.102255e+00f,
                1.296306e+00f,  1.264670e+00f,  1.233034e+00f,  1.201398e+00f,  1.169762e+00f,  1.138126e+00f,  1.106490e+00f,
                1.301267e+00f,  1.269510e+00f,  1.237753e+00f,  1.205996e+00f,  1.174239e+00f,  1.142482e+00f,  1.110725e+00f,
                1.306228e+00f,  1.274350e+00f,  1.242472e+00f,  1.210594e+00f,  1.178716e+00f,  1.146838e+00f,  1.114960e+00f,
                1.311189e+00f,  1.279190e+00f,  1.247191e+00f,  1.215192e+00f,  1.183193e+00f,  1.151194e+00f,  1.119195e+00f,
                1.316150e+00f,  1.284030e+00f,  1.251910e+00f,  1.219790e+00f,  1.187670e+00f,  1.155550e+00f,  1.123430e+00f,
                1.321111e+00f,  1.288870e+00f,  1.256629e+00f,  1.224388e+00f,  1.192147e+00f,  1.159906e+00f,  1.127665e+00f,
                1.326072e+00f,  1.293710e+00f,  1.261348e+00f,  1.228986e+00f,  1.196624e+00f,  1.164262e+00f,  1.131900e+00f,
                1.331033e+00f,  1.298550e+00f,  1.266067e+00f,  1.233584e+00f,  1.201101e+00f,  1.168618e+00f,  1.136135e+00f,
                1.335994e+00f,  1.303390e+00f,  1.270786e+00f,  1.238182e+00f,  1.205578e+00f,  1.172974e+00f,  1.140370e+00f,
                1.340955e+00f,  1.308230e+00f,  1.275505e+00f,  1.242780e+00f,  1.210055e+00f,  1.177330e+00f,  1.144605e+00f,
                1.345916e+00f,  1.313070e+00f,  1.280224e+00f,  1.247378e+00f,  1.214532e+00f,  1.181686e+00f,  1.148840e+00f,
                1.350877e+00f,  1.317910e+00f,  1.284943e+00f,  1.251976e+00f,  1.219009e+00f,  1.186042e+00f,  1.153075e+00f,
                1.355838e+00f,  1.322750e+00f,  1.289662e+00f,  1.256574e+00f,  1.223486e+00f,  1.190398e+00f,  1.157310e+00f,
                1.360799e+00f,  1.327590e+00f,  1.294381e+00f,  1.261172e+00f,  1.227963e+00f,  1.194754e+00f,  1.161545e+00f,
                1.365760e+00f,  1.332430e+00f,  1.299100e+00f,  1.265770e+00f,  1.232440e+00f,  1.199110e+00f,  1.165780e+00f,
                1.370721e+00f,  1.337270e+00f,  1.303819e+00f,  1.270368e+00f,  1.236917e+00f,  1.203466e+00f,  1.170015e+00f,
                1.375682e+00f,  1.342110e+00f,  1.308538e+00f,  1.274966e+00f,  1.241394e+00f,  1.207822e+00f,  1.174250e+00f,
                1.380643e+00f,  1.346950e+00f,  1.313257e+00f,  1.279564e+00f,  1.245871e+00f,  1.212178e+00f,  1.178485e+00f,
                1.385604e+00f,  1.351790e+00f,  1.317976e+00f,  1.284162e+00f,  1.250348e+00f,  1.216534e+00f,  1.182720e+00f,
                1.390565e+00f,  1.356630e+00f,  1.322695e+00f,  1.288760e+00f,  1.254825e+00f,  1.220890e+00f,  1.186955e+00f,
                1.395526e+00f,  1.361470e+00f,  1.327414e+00f,  1.293358e+00f,  1.259302e+00f,  1.225246e+00f,  1.191190e+00f,
                1.400487e+00f,  1.366310e+00f,  1.332133e+00f,  1.297956e+00f,  1.263779e+00f,  1.229602e+00f,  1.195425e+00f,
                1.405448e+00f,  1.371150e+00f,  1.336852e+00f,  1.302554e+00f,  1.268256e+00f,  1.233958e+00f,  1.199660e+00f,
                1.410409e+00f,  1.375990e+00f,  1.341571e+00f,  1.307152e+00f,  1.272733e+00f,  1.238314e+00f,  1.203895e+00f,
                1.415370e+00f,  1.380830e+00f,  1.346290e+00f,  1.311750e+00f,  1.277210e+00f,  1.242670e+00f,  1.208130e+00f,
                1.420331e+00f,  1.385670e+00f,  1.351009e+00f,  1.316348e+00f,  1.281687e+00f,  1.247026e+00f,  1.212365e+00f,
                1.425292e+00f,  1.390510e+00f,  1.355728e+00f,  1.320946e+00f,  1.286164e+00f,  1.251382e+00f,  1.216600e+00f,
                1.430253e+00f,  1.395350e+00f,  1.360447e+00f,  1.325544e+00f,  1.290641e+00f,  1.255738e+00f,  1.220835e+00f,
                1.435214e+00f,  1.400190e+00f,  1.365166e+00f,  1.330142e+00f,  1.295118e+00f,  1.260094e+00f,  1.225070e+00f,
                1.440175e+00f,  1.405030e+00f,  1.369885e+00f,  1.334740e+00f,  1.299595e+00f,  1.264450e+00f,  1.229305e+00f,
                1.445136e+00f,  1.409870e+00f,  1.374604e+00f,  1.339338e+00f,  1.304072e+00f,  1.268806e+00f,  1.233540e+00f,
                1.450097e+00f,  1.414710e+00f,  1.379323e+00f,  1.343936e+00f,  1.308549e+00f,  1.273162e+00f,  1.237775e+00f,
                1.455058e+00f,  1.419550e+00f,  1.384042e+00f,  1.348534e+00f,  1.313026e+00f,  1.277518e+00f,  1.242010e+00f,
                1.460019e+00f,  1.424390e+00f,  1.388761e+00f,  1.353132e+00f,  1.317503e+00f,  1.281874e+00f,  1.246245e+00f,
                1.464980e+00f,  1.429230e+00f,  1.393480e+00f,  1.357730e+00f,  1.321980e+00f,  1.286230e+00f,  1.250480e+00f,
                1.469941e+00f,  1.434070e+00f,  1.398199e+00f,  1.362328e+00f,  1.326457e+00f,  1.290586e+00f,  1.254715e+00f,
                1.474902e+00f,  1.438910e+00f,  1.402918e+00f,  1.366926e+00f,  1.330934e+00f,  1.294942e+00f,  1.258950e+00f,
                1.479863e+00f,  1.443750e+00f,  1.407637e+00f,  1.371524e+00f,  1.335411e+00f,  1.299298e+00f,  1.263185e+00f,
                1.484824e+00f,  1.448590e+00f,  1.412356e+00f,  1.376122e+00f,  1.339888e+00f,  1.303654e+00f,  1.267420e+00f,
                1.489785e+00f,  1.453430e+00f,  1.417075e+00f,  1.380720e+00f,  1.344365e+00f,  1.308010e+00f,  1.271655e+00f,
                1.494746e+00f,  1.458270e+00f,  1.421794e+00f,  1.385318e+00f,  1.348842e+00f,  1.312366e+00f,  1.275890e+00f,
                1.499707e+00f,  1.463110e+00f,  1.426513e+00f,  1.389916e+00f,  1.353319e+00f,  1.316722e+00f,  1.280125e+00f,
                1.504668e+00f,  1.467950e+00f,  1.431232e+00f,  1.394514e+00f,  1.357796e+00f,  1.321078e+00f,  1.284360e+00f,
                1.509629e+00f,  1.472790e+00f,  1.435951e+00f,  1.399112e+00f,  1.362273e+00f,  1.325434e+00f,  1.288595e+00f,
                1.514590e+00f,  1.477630e+00f,  1.440670e+00f,  1.403710e+00f,  1.366750e+00f,  1.329790e+00f,  1.292830e+00f,
                1.519551e+00f,  1.482470e+00f,  1.445389e+00f,  1.408308e+00f,  1.371227e+00f,  1.334146e+00f,  1.297065e+00f,
                1.524512e+00f,  1.487310e+00f,  1.450108e+00f,  1.412906e+00f,  1.375704e+00f,  1.338502e+00f,  1.301300e+00f,
                1.529473e+00f,  1.492150e+00f,  1.454827e+00f,  1.417504e+00f,  1.380181e+00f,  1.342858e+00f,  1.305535e+00f,
                1.534434e+00f,  1.496990e+00f,  1.459546e+00f,  1.422102e+00f,  1.384658e+00f,  1.347214e+00f,  1.309770e+00f,
                1.539395e+00f,  1.501830e+00f,  1.464265e+00f,  1.426700e+00f,  1.389135e+00f,  1.351570e+00f,  1.314005e+00f,
                1.544356e+00f,  1.506670e+00f,  1.468984e+00f,  1.431298e+00f,  1.393612e+00f,  1.355926e+00f,  1.318240e+00f,
                1.549317e+00f,  1.511510e+00f,  1.473703e+00f,  1.435896e+00f,  1.398089e+00f,  1.360282e+00f,  1.322475e+00f,
                1.554278e+00f,  1.516350e+00f,  1.478422e+00f,  1.440494e+00f,  1.402566e+00f,  1.364638e+00f,  1.326710e+00f,
                1.559239e+00f,  1.521190e+00f,  1.483141e+00f,  1.445092e+00f,  1.407043e+00f,  1.368994e+00f,  1.330945e+00f,
                1.564200e+00f,  1.526030e+00f,  1.487860e+00f,  1.449690e+00f,  1.411520e+00f,  1.373350e+00f,  1.335180e+00f,
                1.569161e+00f,  1.530870e+00f,  1.492579e+00f,  1.454288e+00f,  1.415997e+00f,  1.377706e+00f,  1.339415e+00f,
                1.574122e+00f,  1.535710e+00f,  1.497298e+00f,  1.458886e+00f,  1.420474e+00f,  1.382062e+00f,  1.343650e+00f,
                1.579083e+00f,  1.540550e+00f,  1.502017e+00f,  1.463484e+00f,  1.424951e+00f,  1.386418e+00f,  1.347885e+00f,
                1.584044e+00f,  1.545390e+00f,  1.506736e+00f,  1.468082e+00f,  1.429428e+00f,  1.390774e+00f,  1.352120e+00f,
                1.589005e+00f,  1.550230e+00f,  1.511455e+00f,  1.472680e+00f,  1.433905e+00f,  1.395130e+00f,  1.356355e+00f,
                1.593966e+00f,  1.555070e+00f,  1.516174e+00f,  1.477278e+00f,  1.438382e+00f,  1.399486e+00f,  1.360590e+00f,
                1.598927e+00f,  1.559910e+00f,  1.520893e+00f,  1.481876e+00f,  1.442859e+00f,  1.403842e+00f,  1.364825e+00f,
                1.603888e+00f,  1.564750e+00f,  1.525612e+00f,  1.486474e+00f,  1.447336e+00f,  1.408198e+00f,  1.369060e+00f,
                1.608849e+00f,  1.569590e+00f,  1.530331e+00f,  1.491072e+00f,  1.451813e+00f,  1.412554e+00f,  1.373295e+00f,
                1.613810e+00f,  1.574430e+00f,  1.535050e+00f,  1.495670e+00f,  1.456290e+00f,  1.416910e+00f,  1.377530e+00f,
                1.618771e+00f,  1.579270e+00f,  1.539769e+00f,  1.500268e+00f,  1.460767e+00f,  1.421266e+00f,  1.381765e+00f,
                1.623732e+00f,  1.584110e+00f,  1.544488e+00f,  1.504866e+00f,  1.465244e+00f,  1.425622e+00f,  1.386000e+00f,
                1.628693e+00f,  1.588950e+00f,  1.549207e+00f,  1.509464e+00f,  1.469721e+00f,  1.429978e+00f,  1.390235e+00f,
                1.633654e+00f,  1.593790e+00f,  1.553926e+00f,  1.514062e+00f,  1.474198e+00f,  1.434334e+00f,  1.394470e+00f,
                1.638615e+00f,  1.598630e+00f,  1.558645e+00f,  1.518660e+00f,  1.478675e+00f,  1.438690e+00f,  1.398705e+00f,
                1.643576e+00f,  1.603470e+00f,  1.563364e+00f,  1.523258e+00f,  1.483152e+00f,  1.443046e+00f,  1.402940e+00f,
                1.648537e+00f,  1.608310e+00f,  1.568083e+00f,  1.527856e+00f,  1.487629e+00f,  1.447402e+00f,  1.407175e+00f,
                1.653498e+00f,  1.613150e+00f,  1.572802e+00f,  1.532454e+00f,  1.492106e+00f,  1.451758e+00f,  1.411410e+00f,
                1.658459e+00f,  1.617990e+00f,  1.577521e+00f,  1.537052e+00f,  1.496583e+00f,  1.456114e+00f,  1.415645e+00f,
                1.663420e+00f,  1.622830e+00f,  1.582240e+00f,  1.541650e+00f,  1.501060e+00f,  1.460470e+00f,  1.419880e+00f,
                1.668381e+00f,  1.627670e+00f,  1.586959e+00f,  1.546248e+00f,  1.505537e+00f,  1.464826e+00f,  1.424115e+00f,
                1.673342e+00f,  1.632510e+00f,  1.591678e+00f,  1.550846e+00f,  1.510014e+00f,  1.469182e+00f,  1.428350e+00f,
                1.678303e+00f,  1.637350e+00f,  1.596397e+00f,  1.555444e+00f,  1.514491e+00f,  1.473538e+00f,  1.432585e+00f,
                1.683264e+00f,  1.642190e+00f,  1.601116e+00f,  1.560042e+00f,  1.518968e+00f,  1.477894e+00f,  1.436820e+00f,
                1.688225e+00f,  1.647030e+00f,  1.605835e+00f,  1.564640e+00f,  1.523445e+00f,  1.482250e+00f,  1.441055e+00f,
                1.693186e+00f,  1.651870e+00f,  1.610554e+00f,  1.569238e+00f,  1.527922e+00f,  1.486606e+00f,  1.445290e+00f,
                1.698147e+00f,  1.656710e+00f,  1.615273e+00f,  1.573836e+00f,  1.532399e+00f,  1.490962e+00f,  1.449525e+00f,
                1.703108e+00f,  1.661550e+00f,  1.619992e+00f,  1.578434e+00f,  1.536876e+00f,  1.495318e+00f,  1.453760e+00f,
                1.708069e+00f,  1.666390e+00f,  1.624711e+00f,  1.583032e+00f,  1.541353e+00f,  1.499674e+00f,  1.457995e+00f,
                1.713030e+00f,  1.671230e+00f,  1.629430e+00f,  1.587630e+00f,  1.545830e+00f,  1.504030e+00f,  1.462230e+00f,
                1.717991e+00f,  1.676070e+00f,  1.634149e+00f,  1.592228e+00f,  1.550307e+00f,  1.508386e+00f,  1.466465e+00f,
                1.722952e+00f,  1.680910e+00f,  1.638868e+00f,  1.596826e+00f,  1.554784e+00f,  1.512742e+00f,  1.470700e+00f,
                1.727913e+00f,  1.685750e+00f,  1.643587e+00f,  1.601424e+00f,  1.559261e+00f,  1.517098e+00f,  1.474935e+00f,
                1.732874e+00f,  1.690590e+00f,  1.648306e+00f,  1.606022e+00f,  1.563738e+00f,  1.521454e+00f,  1.479170e+00f,
                1.737835e+00f,  1.695430e+00f,  1.653025e+00f,  1.610620e+00f,  1.568215e+00f,  1.525810e+00f,  1.483405e+00f,
                1.742796e+00f,  1.700270e+00f,  1.657744e+00f,  1.615218e+00f,  1.572692e+00f,  1.530166e+00f,  1.487640e+00f,
                1.747757e+00f,  1.705110e+00f,  1.662463e+00f,  1.619816e+00f,  1.577169e+00f,  1.534522e+00f,  1.491875e+00f,
                1.752718e+00f,  1.709950e+00f,  1.667182e+00f,  1.624414e+00f,  1.581646e+00f,  1.538878e+00f,  1.496110e+00f,
                1.757679e+00f,  1.714790e+00f,  1.671901e+00f,  1.629012e+00f,  1.586123e+00f,  1.543234e+00f,  1.500345e+00f,
                1.762640e+00f,  1.719630e+00f,  1.676620e+00f,  1.633610e+00f,  1.590600e+00f,  1.547590e+00f,  1.504580e+00f,
                1.767601e+00f,  1.724470e+00f,  1.681339e+00f,  1.638208e+00f,  1.595077e+00f,  1.551946e+00f,  1.508815e+00f,
                1.772562e+00f,  1.729310e+00f,  1.686058e+00f,  1.642806e+00f,  1.599554e+00f,  1.556302e+00f,  1.513050e+00f,
                1.777523e+00f,  1.734150e+00f,  1.690777e+00f,  1.647404e+00f,  1.604031e+00f,  1.560658e+00f,  1.517285e+00f,
                1.782484e+00f,  1.738990e+00f,  1.695496e+00f,  1.652002e+00f,  1.608508e+00f,  1.565014e+00f,  1.521520e+00f,
                1.787445e+00f,  1.743830e+00f,  1.700215e+00f,  1.656600e+00f,  1.612985e+00f,  1.569370e+00f,  1.525755e+00f,
                1.792406e+00f,  1.748670e+00f,  1.704934e+00f,  1.661198e+00f,  1.617462e+00f,  1.573726e+00f,  1.529990e+00f,
                1.797367e+00f,  1.753510e+00f,  1.709653e+00f,  1.665796e+00f,  1.621939e+00f,  1.578082e+00f,  1.534225e+00f,
                1.802328e+00f,  1.758350e+00f,  1.714372e+00f,  1.670394e+00f,  1.626416e+00f,  1.582438e+00f,  1.538460e+00f,
                1.807289e+00f,  1.763190e+00f,  1.719091e+00f,  1.674992e+00f,  1.630893e+00f,  1.586794e+00f,  1.542695e+00f,
                1.812250e+00f,  1.768030e+00f,  1.723810e+00f,  1.679590e+00f,  1.635370e+00f,  1.591150e+00f,  1.546930e+00f,
                1.817211e+00f,  1.772870e+00f,  1.728529e+00f,  1.684188e+00f,  1.639847e+00f,  1.595506e+00f,  1.551165e+00f,
                1.822172e+00f,  1.777710e+00f,  1.733248e+00f,  1.688786e+00f,  1.644324e+00f,  1.599862e+00f,  1.555400e+00f,
                1.827133e+00f,  1.782550e+00f,  1.737967e+00f,  1.693384e+00f,  1.648801e+00f,  1.604218e+00f,  1.559635e+00f,
                1.832094e+00f,  1.787390e+00f,  1.742686e+00f,  1.697982e+00f,  1.653278e+00f,  1.608574e+00f,  1.563870e+00f,
                1.837055e+00f,  1.792230e+00f,  1.747405e+00f,  1.702580e+00f,  1.657755e+00f,  1.612930e+00f,  1.568105e+00f,
                1.842016e+00f,  1.797070e+00f,  1.752124e+00f,  1.707178e+00f,  1.662232e+00f,  1.617286e+00f,  1.572340e+00f,
                1.846977e+00f,  1.801910e+00f,  1.756843e+00f,  1.711776e+00f,  1.666709e+00f,  1.621642e+00f,  1.576575e+00f,
                1.851938e+00f,  1.806750e+00f,  1.761562e+00f,  1.716374e+00f,  1.671186e+00f,  1.625998e+00f,  1.580810e+00f,
                1.856899e+00f,  1.811590e+00f,  1.766281e+00f,  1.720972e+00f,  1.675663e+00f,  1.630354e+00f,  1.585045e+00f,
                1.861860e+00f,  1.816430e+00f,  1.771000e+00f,  1.725570e+00f,  1.680140e+00f,  1.634710e+00f,  1.589280e+00f,
                1.866821e+00f,  1.821270e+00f,  1.775719e+00f,  1.730168e+00f,  1.684617e+00f,  1.639066e+00f,  1.593515e+00f,
                1.871782e+00f,  1.826110e+00f,  1.780438e+00f,  1.734766e+00f,  1.689094e+00f,  1.643422e+00f,  1.597750e+00f,
                1.876743e+00f,  1.830950e+00f,  1.785157e+00f,  1.739364e+00f,  1.693571e+00f,  1.647778e+00f,  1.601985e+00f,
                1.881704e+00f,  1.835790e+00f,  1.789876e+00f,  1.743962e+00f,  1.698048e+00f,  1.652134e+00f,  1.606220e+00f,
                1.886665e+00f,  1.840630e+00f,  1.794595e+00f,  1.748560e+00f,  1.702525e+00f,  1.656490e+00f,  1.610455e+00f,
                1.891626e+00f,  1.845470e+00f,  1.799314e+00f,  1.753158e+00f,  1.707002e+00f,  1.660846e+00f,  1.614690e+00f,
                1.896587e+00f,  1.850310e+00f,  1.804033e+00f,  1.757756e+00f,  1.711479e+00f,  1.665202e+00f,  1.618925e+00f,
                1.901548e+00f,  1.855150e+00f,  1.808752e+00f,  1.762354e+00f,  1.715956e+00f,  1.669558e+00f,  1.623160e+00f,
                1.906509e+00f,  1.859990e+00f,  1.813471e+00f,  1.766952e+00f,  1.720433e+00f,  1.673914e+00f,  1.627395e+00f,
                1.911470e+00f,  1.864830e+00f,  1.818190e+00f,  1.771550e+00f,  1.724910e+00f,  1.678270e+00f,  1.631630e+00f,
                1.916431e+00f,  1.869670e+00f,  1.822909e+00f,  1.776148e+00f,  1.729387e+00f,  1.682626e+00f,  1.635865e+00f,
                1.921392e+00f,  1.874510e+00f,  1.827628e+00f,  1.780746e+00f,  1.733864e+00f,  1.686982e+00f,  1.640100e+00f,
                1.926353e+00f,  1.879350e+00f,  1.832347e+00f,  1.785344e+00f,  1.738341e+00f,  1.691338e+00f,  1.644335e+00f,
                1.931314e+00f,  1.884190e+00f,  1.837066e+00f,  1.789942e+00f,  1.742818e+00f,  1.695694e+00f,  1.648570e+00f,
                1.936275e+00f,  1.889030e+00f,  1.841785e+00f,  1.794540e+00f,  1.747295e+00f,  1.700050e+00f,  1.652805e+00f,
                1.941236e+00f,  1.893870e+00f,  1.846504e+00f,  1.799138e+00f,  1.751772e+00f,  1.704406e+00f,  1.657040e+00f,
                1.946197e+00f,  1.898710e+00f,  1.851223e+00f,  1.803736e+00f,  1.756249e+00f,  1.708762e+00f,  1.661275e+00f,
                1.951158e+00f,  1.903550e+00f,  1.855942e+00f,  1.808334e+00f,  1.760726e+00f,  1.713118e+00f,  1.665510e+00f,
                1.956119e+00f,  1.908390e+00f,  1.860661e+00f,  1.812932e+00f,  1.765203e+00f,  1.717474e+00f,  1.669745e+00f,
                1.961080e+00f,  1.913230e+00f,  1.865380e+00f,  1.817530e+00f,  1.769680e+00f,  1.721830e+00f,  1.673980e+00f,
                1.966041e+00f,  1.918070e+00f,  1.870099e+00f,  1.822128e+00f,  1.774157e+00f,  1.726186e+00f,  1.678215e+00f,
                1.971002e+00f,  1.922910e+00f,  1.874818e+00f,  1.826726e+00f,  1.778634e+00f,  1.730542e+00f,  1.682450e+00f,
                1.975963e+00f,  1.927750e+00f,  1.879537e+00f,  1.831324e+00f,  1.783111e+00f,  1.734898e+00f,  1.686685e+00f,
                1.980924e+00f,  1.932590e+00f,  1.884256e+00f,  1.835922e+00f,  1.787588e+00f,  1.739254e+00f,  1.690920e+00f,
                1.985885e+00f,  1.937430e+00f,  1.888975e+00f,  1.840520e+00f,  1.792065e+00f,  1.743610e+00f,  1.695155e+00f,
                1.990846e+00f,  1.942270e+00f,  1.893694e+00f,  1.845118e+00f,  1.796542e+00f,  1.747966e+00f,  1.699390e+00f,
                1.995807e+00f,  1.947110e+00f,  1.898413e+00f,  1.849716e+00f,  1.801019e+00f,  1.752322e+00f,  1.703625e+00f,
                2.000768e+00f,  1.951950e+00f,  1.903132e+00f,  1.854314e+00f,  1.805496e+00f,  1.756678e+00f,  1.707860e+00f,
                2.005729e+00f,  1.956790e+00f,  1.907851e+00f,  1.858912e+00f,  1.809973e+00f,  1.761034e+00f,  1.712095e+00f,
                2.010690e+00f,  1.961630e+00f,  1.912570e+00f,  1.863510e+00f,  1.814450e+00f,  1.765390e+00f,  1.716330e+00f,
                2.015651e+00f,  1.966470e+00f,  1.917289e+00f,  1.868108e+00f,  1.818927e+00f,  1.769746e+00f,  1.720565e+00f,
                2.020612e+00f,  1.971310e+00f,  1.922008e+00f,  1.872706e+00f,  1.823404e+00f,  1.774102e+00f,  1.724800e+00f,
                2.025573e+00f,  1.976150e+00f,  1.926727e+00f,  1.877304e+00f,  1.827881e+00f,  1.778458e+00f,  1.729035e+00f,
                2.030534e+00f,  1.980990e+00f,  1.931446e+00f,  1.881902e+00f,  1.832358e+00f,  1.782814e+00f,  1.733270e+00f,
                2.035495e+00f,  1.985830e+00f,  1.936165e+00f,  1.886500e+00f,  1.836835e+00f,  1.787170e+00f,  1.737505e+00f,
                2.040456e+00f,  1.990670e+00f,  1.940884e+00f,  1.891098e+00f,  1.841312e+00f,  1.791526e+00f,  1.741740e+00f,
                2.045417e+00f,  1.995510e+00f,  1.945603e+00f,  1.895696e+00f,  1.845789e+00f,  1.795882e+00f,  1.745975e+00f,
                2.050378e+00f,  2.000350e+00f,  1.950322e+00f,  1.900294e+00f,  1.850266e+00f,  1.800238e+00f,  1.750210e+00f,
                2.055339e+00f,  2.005190e+00f,  1.955041e+00f,  1.904892e+00f,  1.854743e+00f,  1.804594e+00f,  1.754445e+00f,
                2.060300e+00f,  2.010030e+00f,  1.959760e+00f,  1.909490e+00f,  1.859220e+00f,  1.808950e+00f,  1.758680e+00f,
                2.065261e+00f,  2.014870e+00f,  1.964479e+00f,  1.914088e+00f,  1.863697e+00f,  1.813306e+00f,  1.762915e+00f,
                2.070222e+00f,  2.019710e+00f,  1.969198e+00f,  1.918686e+00f,  1.868174e+00f,  1.817662e+00f,  1.767150e+00f,
                2.075183e+00f,  2.024550e+00f,  1.973917e+00f,  1.923284e+00f,  1.872651e+00f,  1.822018e+00f,  1.771385e+00f,
                2.080144e+00f,  2.029390e+00f,  1.978636e+00f,  1.927882e+00f,  1.877128e+00f,  1.826374e+00f,  1.775620e+00f,
                2.085105e+00f,  2.034230e+00f,  1.983355e+00f,  1.932480e+00f,  1.881605e+00f,  1.830730e+00f,  1.779855e+00f,
                2.090066e+00f,  2.039070e+00f,  1.988074e+00f,  1.937078e+00f,  1.886082e+00f,  1.835086e+00f,  1.784090e+00f,
                2.095027e+00f,  2.043910e+00f,  1.992793e+00f,  1.941676e+00f,  1.890559e+00f,  1.839442e+00f,  1.788325e+00f,
                2.099988e+00f,  2.048750e+00f,  1.997512e+00f,  1.946274e+00f,  1.895036e+00f,  1.843798e+00f,  1.792560e+00f,
                2.104949e+00f,  2.053590e+00f,  2.002231e+00f,  1.950872e+00f,  1.899513e+00f,  1.848154e+00f,  1.796795e+00f,
                2.109910e+00f,  2.058430e+00f,  2.006950e+00f,  1.955470e+00f,  1.903990e+00f,  1.852510e+00f,  1.801030e+00f,
                2.114871e+00f,  2.063270e+00f,  2.011669e+00f,  1.960068e+00f,  1.908467e+00f,  1.856866e+00f,  1.805265e+00f,
                2.119832e+00f,  2.068110e+00f,  2.016388e+00f,  1.964666e+00f,  1.912944e+00f,  1.861222e+00f,  1.809500e+00f,
                2.124793e+00f,  2.072950e+00f,  2.021107e+00f,  1.969264e+00f,  1.917421e+00f,  1.865578e+00f,  1.813735e+00f,
                2.129754e+00f,  2.077790e+00f,  2.025826e+00f,  1.973862e+00f,  1.921898e+00f,  1.869934e+00f,  1.817970e+00f,
                2.134715e+00f,  2.082630e+00f,  2.030545e+00f,  1.978460e+00f,  1.926375e+00f,  1.874290e+00f,  1.822205e+00f,
                2.139676e+00f,  2.087470e+00f,  2.035264e+00f,  1.983058e+00f,  1.930852e+00f,  1.878646e+00f,  1.826440e+00f,
                2.144637e+00f,  2.092310e+00f,  2.039983e+00f,  1.987656e+00f,  1.935329e+00f,  1.883002e+00f,  1.830675e+00f,
                2.149598e+00f,  2.097150e+00f,  2.044702e+00f,  1.992254e+00f,  1.939806e+00f,  1.887358e+00f,  1.834910e+00f,
                2.154559e+00f,  2.101990e+00f,  2.049421e+00f,  1.996852e+00f,  1.944283e+00f,  1.891714e+00f,  1.839145e+00f,
                2.159520e+00f,  2.106830e+00f,  2.054140e+00f,  2.001450e+00f,  1.948760e+00f,  1.896070e+00f,  1.843380e+00f,
                2.164481e+00f,  2.111670e+00f,  2.058859e+00f,  2.006048e+00f,  1.953237e+00f,  1.900426e+00f,  1.847615e+00f,
                2.169442e+00f,  2.116510e+00f,  2.063578e+00f,  2.010646e+00f,  1.957714e+00f,  1.904782e+00f,  1.851850e+00f,
                2.174403e+00f,  2.121350e+00f,  2.068297e+00f,  2.015244e+00f,  1.962191e+00f,  1.909138e+00f,  1.856085e+00f,
                2.179364e+00f,  2.126190e+00f,  2.073016e+00f,  2.019842e+00f,  1.966668e+00f,  1.913494e+00f,  1.860320e+00f,
                2.184325e+00f,  2.131030e+00f,  2.077735e+00f,  2.024440e+00f,  1.971145e+00f,  1.917850e+00f,  1.864555e+00f,
                2.189286e+00f,  2.135870e+00f,  2.082454e+00f,  2.029038e+00f,  1.975622e+00f,  1.922206e+00f,  1.868790e+00f,
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{inwidth},{inheight},{batch}");
        }
    }
}
