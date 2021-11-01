using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class DeconvolutionTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                    foreach (int outchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                int outwidth = inwidth - kwidth + 1;

                                float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] wval = (new float[kwidth * inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                Map1D y = new(outchannels, outwidth, batch, yval);
                                Filter1D w = new(inchannels, outchannels, kwidth, wval);

                                Map1D x = Reference(y, w, inwidth, kwidth);

                                OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth, batch), yval);
                                OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels, outchannels, kwidth), wval);

                                OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch));

                                Deconvolution ope = new(outwidth, outchannels, inchannels, kwidth, batch);

                                ope.Execute(y_tensor, w_tensor, x_tensor);

                                float[] x_expect = x.ToArray();
                                float[] x_actual = x_tensor.State.Value;

                                CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                                CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                AssertError.Tolerance(x_expect, x_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{kwidth},{inwidth},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");
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
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                    foreach (int outchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                int outwidth = inwidth - kwidth + 1;

                                float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] wval = (new float[kwidth * inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                Map1D y = new(outchannels, outwidth, batch, yval);
                                Filter1D w = new(inchannels, outchannels, kwidth, wval);

                                Map1D x = Reference(y, w, inwidth, kwidth);

                                OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth, batch), yval);
                                OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels, outchannels, kwidth), wval);

                                OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch));

                                Deconvolution ope = new(outwidth, outchannels, inchannels, kwidth, batch);

                                ope.Execute(y_tensor, w_tensor, x_tensor);

                                float[] x_expect = x.ToArray();
                                float[] x_actual = x_tensor.State.Value;

                                CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                                CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{kwidth},{inwidth},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");
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
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                    foreach (int outchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                int outwidth = inwidth - kwidth + 1;

                                float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] wval = (new float[kwidth * inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                Map1D y = new(outchannels, outwidth, batch, yval);
                                Filter1D w = new(inchannels, outchannels, kwidth, wval);

                                Map1D x = Reference(y, w, inwidth, kwidth);

                                OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth, batch), yval);
                                OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels, outchannels, kwidth), wval);

                                OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch));

                                Deconvolution ope = new(outwidth, outchannels, inchannels, kwidth, batch);

                                ope.Execute(y_tensor, w_tensor, x_tensor);

                                float[] x_expect = x.ToArray();
                                float[] x_actual = x_tensor.State.Value;

                                CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                                CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                AssertError.Tolerance(x_expect, x_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{kwidth},{inwidth},{batch}");

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
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            float max_err = 0;

            Random random = new(1234);

            int batch = 3;
            int inchannels = 49, outchannels = 50;
            int kwidth = 5;
            int inwidth = 125;
            int outwidth = inwidth - kwidth + 1;

            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * inchannels * outchannels]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map1D y = new(outchannels, outwidth, batch, yval);
            Filter1D w = new(inchannels, outchannels, kwidth, wval);

            Map1D x = Reference(y, w, inwidth, kwidth);

            OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth, batch), yval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels, outchannels, kwidth), wval);

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch));

            Deconvolution ope = new(outwidth, outchannels, inchannels, kwidth, batch);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            float[] x_expect = x.ToArray();
            float[] x_actual = x_tensor.State.Value;

            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 512, inchannels = 31, outchannels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels, outchannels, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth));

            Deconvolution ope = new(outwidth, outchannels, inchannels, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/deconvolution_1d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 512, inchannels = 31, outchannels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels, outchannels, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth));

            Deconvolution ope = new(outwidth, outchannels, inchannels, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/deconvolution_1d_ffp.nvvp");
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

            int inwidth = 512, inchannels = 31, outchannels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels, outchannels, ksize));

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth));

            Deconvolution ope = new(outwidth, outchannels, inchannels, ksize);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/deconvolution_1d_cudnn.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map1D Reference(Map1D y, Filter1D w, int inw, int kwidth) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int outw = inw - kwidth + 1;

            if (y.Width != outw) {
                throw new ArgumentException("mismatch shape");
            }

            Map1D x = new(inchannels, inw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        for (int outch = 0; outch < outchannels; outch++) {
                            double v = y[outch, ox, th];

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
            int inchannels = 7, outchannels = 11, kwidth = 3, inwidth = 13, batch = 2;
            int outwidth = inwidth - kwidth + 1;

            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map1D y = new(outchannels, outwidth, batch, yval);
            Filter1D w = new(inchannels, outchannels, kwidth, wval);

            Map1D x = Reference(y, w, inwidth, kwidth);

            float[] x_expect = {
                9.955000000e-03f, 9.900000000e-03f, 9.845000000e-03f, 9.790000000e-03f, 9.735000000e-03f, 9.680000000e-03f, 9.625000000e-03f,
                3.927000000e-02f, 3.903900000e-02f, 3.880800000e-02f, 3.857700000e-02f, 3.834600000e-02f, 3.811500000e-02f, 3.788400000e-02f,
                7.862800000e-02f, 7.810000000e-02f, 7.757200000e-02f, 7.704400000e-02f, 7.651600000e-02f, 7.598800000e-02f, 7.546000000e-02f,
                1.214620000e-01f, 1.205710000e-01f, 1.196800000e-01f, 1.187890000e-01f, 1.178980000e-01f, 1.170070000e-01f, 1.161160000e-01f,
                1.642960000e-01f, 1.630420000e-01f, 1.617880000e-01f, 1.605340000e-01f, 1.592800000e-01f, 1.580260000e-01f, 1.567720000e-01f,
                2.071300000e-01f, 2.055130000e-01f, 2.038960000e-01f, 2.022790000e-01f, 2.006620000e-01f, 1.990450000e-01f, 1.974280000e-01f,
                2.499640000e-01f, 2.479840000e-01f, 2.460040000e-01f, 2.440240000e-01f, 2.420440000e-01f, 2.400640000e-01f, 2.380840000e-01f,
                2.927980000e-01f, 2.904550000e-01f, 2.881120000e-01f, 2.857690000e-01f, 2.834260000e-01f, 2.810830000e-01f, 2.787400000e-01f,
                3.356320000e-01f, 3.329260000e-01f, 3.302200000e-01f, 3.275140000e-01f, 3.248080000e-01f, 3.221020000e-01f, 3.193960000e-01f,
                3.784660000e-01f, 3.753970000e-01f, 3.723280000e-01f, 3.692590000e-01f, 3.661900000e-01f, 3.631210000e-01f, 3.600520000e-01f,
                4.213000000e-01f, 4.178680000e-01f, 4.144360000e-01f, 4.110040000e-01f, 4.075720000e-01f, 4.041400000e-01f, 4.007080000e-01f,
                1.946340000e-01f, 1.922250000e-01f, 1.898160000e-01f, 1.874070000e-01f, 1.849980000e-01f, 1.825890000e-01f, 1.801800000e-01f,
                5.109500000e-02f, 4.983000000e-02f, 4.856500000e-02f, 4.730000000e-02f, 4.603500000e-02f, 4.477000000e-02f, 4.350500000e-02f,
                2.695000000e-01f, 2.681140000e-01f, 2.667280000e-01f, 2.653420000e-01f, 2.639560000e-01f, 2.625700000e-01f, 2.611840000e-01f,
                4.558730000e-01f, 4.529800000e-01f, 4.500870000e-01f, 4.471940000e-01f, 4.443010000e-01f, 4.414080000e-01f, 4.385150000e-01f,
                5.498020000e-01f, 5.452810000e-01f, 5.407600000e-01f, 5.362390000e-01f, 5.317180000e-01f, 5.271970000e-01f, 5.226760000e-01f,
                5.926360000e-01f, 5.877520000e-01f, 5.828680000e-01f, 5.779840000e-01f, 5.731000000e-01f, 5.682160000e-01f, 5.633320000e-01f,
                6.354700000e-01f, 6.302230000e-01f, 6.249760000e-01f, 6.197290000e-01f, 6.144820000e-01f, 6.092350000e-01f, 6.039880000e-01f,
                6.783040000e-01f, 6.726940000e-01f, 6.670840000e-01f, 6.614740000e-01f, 6.558640000e-01f, 6.502540000e-01f, 6.446440000e-01f,
                7.211380000e-01f, 7.151650000e-01f, 7.091920000e-01f, 7.032190000e-01f, 6.972460000e-01f, 6.912730000e-01f, 6.853000000e-01f,
                7.639720000e-01f, 7.576360000e-01f, 7.513000000e-01f, 7.449640000e-01f, 7.386280000e-01f, 7.322920000e-01f, 7.259560000e-01f,
                8.068060000e-01f, 8.001070000e-01f, 7.934080000e-01f, 7.867090000e-01f, 7.800100000e-01f, 7.733110000e-01f, 7.666120000e-01f,
                8.496400000e-01f, 8.425780000e-01f, 8.355160000e-01f, 8.284540000e-01f, 8.213920000e-01f, 8.143300000e-01f, 8.072680000e-01f,
                8.924740000e-01f, 8.850490000e-01f, 8.776240000e-01f, 8.701990000e-01f, 8.627740000e-01f, 8.553490000e-01f, 8.479240000e-01f,
                4.062630000e-01f, 4.011920000e-01f, 3.961210000e-01f, 3.910500000e-01f, 3.859790000e-01f, 3.809080000e-01f, 3.758370000e-01f,
                1.056660000e-01f, 1.030700000e-01f, 1.004740000e-01f, 9.787800000e-02f, 9.528200000e-02f, 9.268600000e-02f, 9.009000000e-02f,
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{kwidth},{inwidth},{batch}");
        }
    }
}
