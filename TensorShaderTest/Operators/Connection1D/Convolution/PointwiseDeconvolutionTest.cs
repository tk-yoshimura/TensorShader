using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class PointwiseDeconvolutionTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                            float[] yval = (new float[inwidth * outchannels * batch]).Select((_, idx) => idx * 1e-4f).ToArray();
                            float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => idx * 1e-4f).Reverse().ToArray();

                            Map1D y = new Map1D(outchannels, inwidth, batch, yval);
                            Filter1D w = new Filter1D(inchannels, outchannels, 1, wval);

                            Map1D x = Reference(y, w);

                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, inwidth, batch), yval);
                            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels), wval);

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch));

                            PointwiseDeconvolution ope = new PointwiseDeconvolution(inwidth, outchannels, inchannels, batch);

                            ope.Execute(y_tensor, w_tensor, x_tensor);

                            float[] x_expect = x.ToArray();
                            float[] x_actual = x_tensor.State;

                            CollectionAssert.AreEqual(yval, y_tensor.State);
                            CollectionAssert.AreEqual(wval, w_tensor.State);

                            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{inwidth},{batch}");

                            Console.WriteLine($"pass: {inchannels},{outchannels},{inwidth},{batch}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inchannels = 31, outchannels = 63;

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, inwidth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));

            PointwiseDeconvolution ope = new PointwiseDeconvolution(inwidth, outchannels, inchannels);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/ptwise_deconvolution_1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);
            
            Cuda.Profiler.Stop();
        }

        public static Map1D Reference(Map1D y, Filter1D w) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int inw = y.Width;

            Map1D x = new Map1D(inchannels, inw, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix = 0; ix < inw; ix++) {
                    for (int outch = 0; outch < outchannels; outch++) {
                        double v = y[outch, ix, th];

                        for (int inch = 0; inch < inchannels; inch++) {
                            x[inch, ix, th] += v * w[inch, outch, 0];
                        }
                    }
                }
            }

            return x;
        }

        public static Map1D Reference2(Map1D y, Filter1D w) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int inw = y.Width;

            Map1D x = new Map1D(inchannels, inw, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix = 0; ix < inw; ix++) {
                    int inch;
                    double[] temp = new double[4];

                    for (inch = 0; inch < inchannels - inchannels % 4; inch += 4) {
                        for (int i = 0; i < 4; i++) {
                            temp[i] = x[inch + i, ix, th];
                        }

                        for (int outch = 0; outch < outchannels; outch++) {
                            double yv = y[outch, ix, th];

                            for (int i = 0; i < 4; i++) {
                                temp[i] += yv * w[inch + i, outch, 0];
                            }
                        }

                        for (int i = 0; i < 4; i++) {
                            x[inch + i, ix, th] = temp[i];
                        }
                    }

                    if (inchannels % 4 != 0) {
                        int sets = inchannels % 4;

                        for (int i = 0; i < sets; i++) {
                            temp[i] = x[inch + i, ix, th];
                        }

                        for (int outch = 0; outch < outchannels; outch++) {
                            double yv = y[outch, ix, th];

                            for (int i = 0; i < sets; i++) {
                                temp[i] += yv * w[inch + i, outch, 0];
                            }
                        }

                        for (int i = 0; i < sets; i++) {
                            x[inch + i, ix, th] = temp[i];
                        }
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, inwidth = 13, batch = 2;

            float[] yval = (new float[inwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map1D y = new Map1D(outchannels, inwidth, batch, yval);
            Filter1D w = new Filter1D(inchannels, outchannels, 1, wval);

            Map1D x = Reference(y, w);

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
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{inwidth},{batch}");
        }
    }
}
