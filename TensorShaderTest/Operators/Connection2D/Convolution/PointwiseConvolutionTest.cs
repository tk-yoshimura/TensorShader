using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class PointwiseConvolutionTest {
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
                                float[] xval = (new float[width * height * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                Map2D x = new(inchannels, width, height, batch, xval);
                                Filter2D w = new(inchannels, outchannels, 1, 1, wval);

                                Map2D y = Reference(x, w);

                                OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, width, height, batch), xval);
                                OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

                                OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, width, height, batch));

                                PointwiseConvolution ope = new(width, height, inchannels, outchannels, batch);

                                ope.Execute(x_tensor, w_tensor, y_tensor);

                                float[] y_expect = y.ToArray();
                                float[] y_actual = y_tensor.State.Value;

                                CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                AssertError.Tolerance(y_expect, y_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{width},{height},{batch}");

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
                                float[] xval = (new float[width * height * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                Map2D x = new(inchannels, width, height, batch, xval);
                                Filter2D w = new(inchannels, outchannels, 1, 1, wval);

                                Map2D y = Reference(x, w);

                                OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, width, height, batch), xval);
                                OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

                                OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, width, height, batch));

                                PointwiseConvolution ope = new(width, height, inchannels, outchannels, batch);

                                ope.Execute(x_tensor, w_tensor, y_tensor);

                                float[] y_expect = y.ToArray();
                                float[] y_actual = y_tensor.State.Value;

                                CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{width},{height},{batch}");

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
                                float[] xval = (new float[width * height * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                Map2D x = new(inchannels, width, height, batch, xval);
                                Filter2D w = new(inchannels, outchannels, 1, 1, wval);

                                Map2D y = Reference(x, w);

                                OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, width, height, batch), xval);
                                OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

                                OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, width, height, batch));

                                PointwiseConvolution ope = new(width, height, inchannels, outchannels, batch);

                                ope.Execute(x_tensor, w_tensor, y_tensor);

                                float[] y_expect = y.ToArray();
                                float[] y_actual = y_tensor.State.Value;

                                CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                AssertError.Tolerance(y_expect, y_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{width},{height},{batch}");

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

            float[] xval = (new float[width * height * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map2D x = new(inchannels, width, height, batch, xval);
            Filter2D w = new(inchannels, outchannels, 1, 1, wval);

            Map2D y = Reference(x, w);

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, width, height, batch), xval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, width, height, batch));

            PointwiseConvolution ope = new(width, height, inchannels, outchannels, batch);

            ope.Execute(x_tensor, w_tensor, y_tensor);

            float[] y_expect = y.ToArray();
            float[] y_actual = y_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{width},{height},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{width},{height},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 512, inheight = 512, inchannels = 31, outchannels = 63;

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, inwidth, inheight));

            PointwiseConvolution ope = new(inwidth, inheight, inchannels, outchannels);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/ptwise_convolution_2d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 512, inheight = 512, inchannels = 31, outchannels = 63;

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, inwidth, inheight));

            PointwiseConvolution ope = new(inwidth, inheight, inchannels, outchannels);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/ptwise_convolution_2d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

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

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor y_tensor = new(Shape.Map2D(outchannels, inwidth, inheight));

            PointwiseConvolution ope = new(inwidth, inheight, inchannels, outchannels);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/ptwise_convolution_2d_cudnn.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map2D Reference(Map2D x, Filter2D w) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, inh = x.Height;

            Map2D y = new(outchannels, inw, inh, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix, iy = 0; iy < inh; iy++) {
                    for (ix = 0; ix < inw; ix++) {
                        for (int outch = 0; outch < outchannels; outch++) {
                            double sum = y[outch, ix, iy, th];

                            for (int inch = 0; inch < inchannels; inch++) {
                                sum += x[inch, ix, iy, th] * w[inch, outch, 0, 0];
                            }

                            y[outch, ix, iy, th] = sum;
                        }
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, inwidth = 13, inheight = 17, batch = 2;

            float[] xval = (new float[batch * inwidth * inheight * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map2D x = new(inchannels, inwidth, inheight, batch, xval);
            Filter2D w = new(inchannels, outchannels, 1, 1, wval);

            Map2D y = Reference(x, w);

            float[] y_expect = {
                1.505000e-03f,  1.358000e-03f,  1.211000e-03f,  1.064000e-03f,  9.170000e-04f,  7.700000e-04f,  6.230000e-04f,  4.760000e-04f,  3.290000e-04f,  1.820000e-04f,  3.500000e-05f,
                5.082000e-03f,  4.592000e-03f,  4.102000e-03f,  3.612000e-03f,  3.122000e-03f,  2.632000e-03f,  2.142000e-03f,  1.652000e-03f,  1.162000e-03f,  6.720000e-04f,  1.820000e-04f,
                8.659000e-03f,  7.826000e-03f,  6.993000e-03f,  6.160000e-03f,  5.327000e-03f,  4.494000e-03f,  3.661000e-03f,  2.828000e-03f,  1.995000e-03f,  1.162000e-03f,  3.290000e-04f,
                1.223600e-02f,  1.106000e-02f,  9.884000e-03f,  8.708000e-03f,  7.532000e-03f,  6.356000e-03f,  5.180000e-03f,  4.004000e-03f,  2.828000e-03f,  1.652000e-03f,  4.760000e-04f,
                1.581300e-02f,  1.429400e-02f,  1.277500e-02f,  1.125600e-02f,  9.737000e-03f,  8.218000e-03f,  6.699000e-03f,  5.180000e-03f,  3.661000e-03f,  2.142000e-03f,  6.230000e-04f,
                1.939000e-02f,  1.752800e-02f,  1.566600e-02f,  1.380400e-02f,  1.194200e-02f,  1.008000e-02f,  8.218000e-03f,  6.356000e-03f,  4.494000e-03f,  2.632000e-03f,  7.700000e-04f,
                2.296700e-02f,  2.076200e-02f,  1.855700e-02f,  1.635200e-02f,  1.414700e-02f,  1.194200e-02f,  9.737000e-03f,  7.532000e-03f,  5.327000e-03f,  3.122000e-03f,  9.170000e-04f,
                2.654400e-02f,  2.399600e-02f,  2.144800e-02f,  1.890000e-02f,  1.635200e-02f,  1.380400e-02f,  1.125600e-02f,  8.708000e-03f,  6.160000e-03f,  3.612000e-03f,  1.064000e-03f,
                3.012100e-02f,  2.723000e-02f,  2.433900e-02f,  2.144800e-02f,  1.855700e-02f,  1.566600e-02f,  1.277500e-02f,  9.884000e-03f,  6.993000e-03f,  4.102000e-03f,  1.211000e-03f,
                3.369800e-02f,  3.046400e-02f,  2.723000e-02f,  2.399600e-02f,  2.076200e-02f,  1.752800e-02f,  1.429400e-02f,  1.106000e-02f,  7.826000e-03f,  4.592000e-03f,  1.358000e-03f,
                3.727500e-02f,  3.369800e-02f,  3.012100e-02f,  2.654400e-02f,  2.296700e-02f,  1.939000e-02f,  1.581300e-02f,  1.223600e-02f,  8.659000e-03f,  5.082000e-03f,  1.505000e-03f,
                4.085200e-02f,  3.693200e-02f,  3.301200e-02f,  2.909200e-02f,  2.517200e-02f,  2.125200e-02f,  1.733200e-02f,  1.341200e-02f,  9.492000e-03f,  5.572000e-03f,  1.652000e-03f,
                4.442900e-02f,  4.016600e-02f,  3.590300e-02f,  3.164000e-02f,  2.737700e-02f,  2.311400e-02f,  1.885100e-02f,  1.458800e-02f,  1.032500e-02f,  6.062000e-03f,  1.799000e-03f,
                4.800600e-02f,  4.340000e-02f,  3.879400e-02f,  3.418800e-02f,  2.958200e-02f,  2.497600e-02f,  2.037000e-02f,  1.576400e-02f,  1.115800e-02f,  6.552000e-03f,  1.946000e-03f,
                5.158300e-02f,  4.663400e-02f,  4.168500e-02f,  3.673600e-02f,  3.178700e-02f,  2.683800e-02f,  2.188900e-02f,  1.694000e-02f,  1.199100e-02f,  7.042000e-03f,  2.093000e-03f,
                5.516000e-02f,  4.986800e-02f,  4.457600e-02f,  3.928400e-02f,  3.399200e-02f,  2.870000e-02f,  2.340800e-02f,  1.811600e-02f,  1.282400e-02f,  7.532000e-03f,  2.240000e-03f,
                5.873700e-02f,  5.310200e-02f,  4.746700e-02f,  4.183200e-02f,  3.619700e-02f,  3.056200e-02f,  2.492700e-02f,  1.929200e-02f,  1.365700e-02f,  8.022000e-03f,  2.387000e-03f,
                6.231400e-02f,  5.633600e-02f,  5.035800e-02f,  4.438000e-02f,  3.840200e-02f,  3.242400e-02f,  2.644600e-02f,  2.046800e-02f,  1.449000e-02f,  8.512000e-03f,  2.534000e-03f,
                6.589100e-02f,  5.957000e-02f,  5.324900e-02f,  4.692800e-02f,  4.060700e-02f,  3.428600e-02f,  2.796500e-02f,  2.164400e-02f,  1.532300e-02f,  9.002000e-03f,  2.681000e-03f,
                6.946800e-02f,  6.280400e-02f,  5.614000e-02f,  4.947600e-02f,  4.281200e-02f,  3.614800e-02f,  2.948400e-02f,  2.282000e-02f,  1.615600e-02f,  9.492000e-03f,  2.828000e-03f,
                7.304500e-02f,  6.603800e-02f,  5.903100e-02f,  5.202400e-02f,  4.501700e-02f,  3.801000e-02f,  3.100300e-02f,  2.399600e-02f,  1.698900e-02f,  9.982000e-03f,  2.975000e-03f,
                7.662200e-02f,  6.927200e-02f,  6.192200e-02f,  5.457200e-02f,  4.722200e-02f,  3.987200e-02f,  3.252200e-02f,  2.517200e-02f,  1.782200e-02f,  1.047200e-02f,  3.122000e-03f,
                8.019900e-02f,  7.250600e-02f,  6.481300e-02f,  5.712000e-02f,  4.942700e-02f,  4.173400e-02f,  3.404100e-02f,  2.634800e-02f,  1.865500e-02f,  1.096200e-02f,  3.269000e-03f,
                8.377600e-02f,  7.574000e-02f,  6.770400e-02f,  5.966800e-02f,  5.163200e-02f,  4.359600e-02f,  3.556000e-02f,  2.752400e-02f,  1.948800e-02f,  1.145200e-02f,  3.416000e-03f,
                8.735300e-02f,  7.897400e-02f,  7.059500e-02f,  6.221600e-02f,  5.383700e-02f,  4.545800e-02f,  3.707900e-02f,  2.870000e-02f,  2.032100e-02f,  1.194200e-02f,  3.563000e-03f,
                9.093000e-02f,  8.220800e-02f,  7.348600e-02f,  6.476400e-02f,  5.604200e-02f,  4.732000e-02f,  3.859800e-02f,  2.987600e-02f,  2.115400e-02f,  1.243200e-02f,  3.710000e-03f,
                9.450700e-02f,  8.544200e-02f,  7.637700e-02f,  6.731200e-02f,  5.824700e-02f,  4.918200e-02f,  4.011700e-02f,  3.105200e-02f,  2.198700e-02f,  1.292200e-02f,  3.857000e-03f,
                9.808400e-02f,  8.867600e-02f,  7.926800e-02f,  6.986000e-02f,  6.045200e-02f,  5.104400e-02f,  4.163600e-02f,  3.222800e-02f,  2.282000e-02f,  1.341200e-02f,  4.004000e-03f,
                1.016610e-01f,  9.191000e-02f,  8.215900e-02f,  7.240800e-02f,  6.265700e-02f,  5.290600e-02f,  4.315500e-02f,  3.340400e-02f,  2.365300e-02f,  1.390200e-02f,  4.151000e-03f,
                1.052380e-01f,  9.514400e-02f,  8.505000e-02f,  7.495600e-02f,  6.486200e-02f,  5.476800e-02f,  4.467400e-02f,  3.458000e-02f,  2.448600e-02f,  1.439200e-02f,  4.298000e-03f,
                1.088150e-01f,  9.837800e-02f,  8.794100e-02f,  7.750400e-02f,  6.706700e-02f,  5.663000e-02f,  4.619300e-02f,  3.575600e-02f,  2.531900e-02f,  1.488200e-02f,  4.445000e-03f,
                1.123920e-01f,  1.016120e-01f,  9.083200e-02f,  8.005200e-02f,  6.927200e-02f,  5.849200e-02f,  4.771200e-02f,  3.693200e-02f,  2.615200e-02f,  1.537200e-02f,  4.592000e-03f,
                1.159690e-01f,  1.048460e-01f,  9.372300e-02f,  8.260000e-02f,  7.147700e-02f,  6.035400e-02f,  4.923100e-02f,  3.810800e-02f,  2.698500e-02f,  1.586200e-02f,  4.739000e-03f,
                1.195460e-01f,  1.080800e-01f,  9.661400e-02f,  8.514800e-02f,  7.368200e-02f,  6.221600e-02f,  5.075000e-02f,  3.928400e-02f,  2.781800e-02f,  1.635200e-02f,  4.886000e-03f,
                1.231230e-01f,  1.113140e-01f,  9.950500e-02f,  8.769600e-02f,  7.588700e-02f,  6.407800e-02f,  5.226900e-02f,  4.046000e-02f,  2.865100e-02f,  1.684200e-02f,  5.033000e-03f,
                1.267000e-01f,  1.145480e-01f,  1.023960e-01f,  9.024400e-02f,  7.809200e-02f,  6.594000e-02f,  5.378800e-02f,  4.163600e-02f,  2.948400e-02f,  1.733200e-02f,  5.180000e-03f,
                1.302770e-01f,  1.177820e-01f,  1.052870e-01f,  9.279200e-02f,  8.029700e-02f,  6.780200e-02f,  5.530700e-02f,  4.281200e-02f,  3.031700e-02f,  1.782200e-02f,  5.327000e-03f,
                1.338540e-01f,  1.210160e-01f,  1.081780e-01f,  9.534000e-02f,  8.250200e-02f,  6.966400e-02f,  5.682600e-02f,  4.398800e-02f,  3.115000e-02f,  1.831200e-02f,  5.474000e-03f,
                1.374310e-01f,  1.242500e-01f,  1.110690e-01f,  9.788800e-02f,  8.470700e-02f,  7.152600e-02f,  5.834500e-02f,  4.516400e-02f,  3.198300e-02f,  1.880200e-02f,  5.621000e-03f,
                1.410080e-01f,  1.274840e-01f,  1.139600e-01f,  1.004360e-01f,  8.691200e-02f,  7.338800e-02f,  5.986400e-02f,  4.634000e-02f,  3.281600e-02f,  1.929200e-02f,  5.768000e-03f,
                1.445850e-01f,  1.307180e-01f,  1.168510e-01f,  1.029840e-01f,  8.911700e-02f,  7.525000e-02f,  6.138300e-02f,  4.751600e-02f,  3.364900e-02f,  1.978200e-02f,  5.915000e-03f,
                1.481620e-01f,  1.339520e-01f,  1.197420e-01f,  1.055320e-01f,  9.132200e-02f,  7.711200e-02f,  6.290200e-02f,  4.869200e-02f,  3.448200e-02f,  2.027200e-02f,  6.062000e-03f,
                1.517390e-01f,  1.371860e-01f,  1.226330e-01f,  1.080800e-01f,  9.352700e-02f,  7.897400e-02f,  6.442100e-02f,  4.986800e-02f,  3.531500e-02f,  2.076200e-02f,  6.209000e-03f,
                1.553160e-01f,  1.404200e-01f,  1.255240e-01f,  1.106280e-01f,  9.573200e-02f,  8.083600e-02f,  6.594000e-02f,  5.104400e-02f,  3.614800e-02f,  2.125200e-02f,  6.356000e-03f,
                1.588930e-01f,  1.436540e-01f,  1.284150e-01f,  1.131760e-01f,  9.793700e-02f,  8.269800e-02f,  6.745900e-02f,  5.222000e-02f,  3.698100e-02f,  2.174200e-02f,  6.503000e-03f,
                1.624700e-01f,  1.468880e-01f,  1.313060e-01f,  1.157240e-01f,  1.001420e-01f,  8.456000e-02f,  6.897800e-02f,  5.339600e-02f,  3.781400e-02f,  2.223200e-02f,  6.650000e-03f,
                1.660470e-01f,  1.501220e-01f,  1.341970e-01f,  1.182720e-01f,  1.023470e-01f,  8.642200e-02f,  7.049700e-02f,  5.457200e-02f,  3.864700e-02f,  2.272200e-02f,  6.797000e-03f,
                1.696240e-01f,  1.533560e-01f,  1.370880e-01f,  1.208200e-01f,  1.045520e-01f,  8.828400e-02f,  7.201600e-02f,  5.574800e-02f,  3.948000e-02f,  2.321200e-02f,  6.944000e-03f,
                1.732010e-01f,  1.565900e-01f,  1.399790e-01f,  1.233680e-01f,  1.067570e-01f,  9.014600e-02f,  7.353500e-02f,  5.692400e-02f,  4.031300e-02f,  2.370200e-02f,  7.091000e-03f,
                1.767780e-01f,  1.598240e-01f,  1.428700e-01f,  1.259160e-01f,  1.089620e-01f,  9.200800e-02f,  7.505400e-02f,  5.810000e-02f,  4.114600e-02f,  2.419200e-02f,  7.238000e-03f,
                1.803550e-01f,  1.630580e-01f,  1.457610e-01f,  1.284640e-01f,  1.111670e-01f,  9.387000e-02f,  7.657300e-02f,  5.927600e-02f,  4.197900e-02f,  2.468200e-02f,  7.385000e-03f,
                1.839320e-01f,  1.662920e-01f,  1.486520e-01f,  1.310120e-01f,  1.133720e-01f,  9.573200e-02f,  7.809200e-02f,  6.045200e-02f,  4.281200e-02f,  2.517200e-02f,  7.532000e-03f,
                1.875090e-01f,  1.695260e-01f,  1.515430e-01f,  1.335600e-01f,  1.155770e-01f,  9.759400e-02f,  7.961100e-02f,  6.162800e-02f,  4.364500e-02f,  2.566200e-02f,  7.679000e-03f,
                1.910860e-01f,  1.727600e-01f,  1.544340e-01f,  1.361080e-01f,  1.177820e-01f,  9.945600e-02f,  8.113000e-02f,  6.280400e-02f,  4.447800e-02f,  2.615200e-02f,  7.826000e-03f,
                1.946630e-01f,  1.759940e-01f,  1.573250e-01f,  1.386560e-01f,  1.199870e-01f,  1.013180e-01f,  8.264900e-02f,  6.398000e-02f,  4.531100e-02f,  2.664200e-02f,  7.973000e-03f,
                1.982400e-01f,  1.792280e-01f,  1.602160e-01f,  1.412040e-01f,  1.221920e-01f,  1.031800e-01f,  8.416800e-02f,  6.515600e-02f,  4.614400e-02f,  2.713200e-02f,  8.120000e-03f,
                2.018170e-01f,  1.824620e-01f,  1.631070e-01f,  1.437520e-01f,  1.243970e-01f,  1.050420e-01f,  8.568700e-02f,  6.633200e-02f,  4.697700e-02f,  2.762200e-02f,  8.267000e-03f,
                2.053940e-01f,  1.856960e-01f,  1.659980e-01f,  1.463000e-01f,  1.266020e-01f,  1.069040e-01f,  8.720600e-02f,  6.750800e-02f,  4.781000e-02f,  2.811200e-02f,  8.414000e-03f,
                2.089710e-01f,  1.889300e-01f,  1.688890e-01f,  1.488480e-01f,  1.288070e-01f,  1.087660e-01f,  8.872500e-02f,  6.868400e-02f,  4.864300e-02f,  2.860200e-02f,  8.561000e-03f,
                2.125480e-01f,  1.921640e-01f,  1.717800e-01f,  1.513960e-01f,  1.310120e-01f,  1.106280e-01f,  9.024400e-02f,  6.986000e-02f,  4.947600e-02f,  2.909200e-02f,  8.708000e-03f,
                2.161250e-01f,  1.953980e-01f,  1.746710e-01f,  1.539440e-01f,  1.332170e-01f,  1.124900e-01f,  9.176300e-02f,  7.103600e-02f,  5.030900e-02f,  2.958200e-02f,  8.855000e-03f,
                2.197020e-01f,  1.986320e-01f,  1.775620e-01f,  1.564920e-01f,  1.354220e-01f,  1.143520e-01f,  9.328200e-02f,  7.221200e-02f,  5.114200e-02f,  3.007200e-02f,  9.002000e-03f,
                2.232790e-01f,  2.018660e-01f,  1.804530e-01f,  1.590400e-01f,  1.376270e-01f,  1.162140e-01f,  9.480100e-02f,  7.338800e-02f,  5.197500e-02f,  3.056200e-02f,  9.149000e-03f,
                2.268560e-01f,  2.051000e-01f,  1.833440e-01f,  1.615880e-01f,  1.398320e-01f,  1.180760e-01f,  9.632000e-02f,  7.456400e-02f,  5.280800e-02f,  3.105200e-02f,  9.296000e-03f,
                2.304330e-01f,  2.083340e-01f,  1.862350e-01f,  1.641360e-01f,  1.420370e-01f,  1.199380e-01f,  9.783900e-02f,  7.574000e-02f,  5.364100e-02f,  3.154200e-02f,  9.443000e-03f,
                2.340100e-01f,  2.115680e-01f,  1.891260e-01f,  1.666840e-01f,  1.442420e-01f,  1.218000e-01f,  9.935800e-02f,  7.691600e-02f,  5.447400e-02f,  3.203200e-02f,  9.590000e-03f,
                2.375870e-01f,  2.148020e-01f,  1.920170e-01f,  1.692320e-01f,  1.464470e-01f,  1.236620e-01f,  1.008770e-01f,  7.809200e-02f,  5.530700e-02f,  3.252200e-02f,  9.737000e-03f,
                2.411640e-01f,  2.180360e-01f,  1.949080e-01f,  1.717800e-01f,  1.486520e-01f,  1.255240e-01f,  1.023960e-01f,  7.926800e-02f,  5.614000e-02f,  3.301200e-02f,  9.884000e-03f,
                2.447410e-01f,  2.212700e-01f,  1.977990e-01f,  1.743280e-01f,  1.508570e-01f,  1.273860e-01f,  1.039150e-01f,  8.044400e-02f,  5.697300e-02f,  3.350200e-02f,  1.003100e-02f,
                2.483180e-01f,  2.245040e-01f,  2.006900e-01f,  1.768760e-01f,  1.530620e-01f,  1.292480e-01f,  1.054340e-01f,  8.162000e-02f,  5.780600e-02f,  3.399200e-02f,  1.017800e-02f,
                2.518950e-01f,  2.277380e-01f,  2.035810e-01f,  1.794240e-01f,  1.552670e-01f,  1.311100e-01f,  1.069530e-01f,  8.279600e-02f,  5.863900e-02f,  3.448200e-02f,  1.032500e-02f,
                2.554720e-01f,  2.309720e-01f,  2.064720e-01f,  1.819720e-01f,  1.574720e-01f,  1.329720e-01f,  1.084720e-01f,  8.397200e-02f,  5.947200e-02f,  3.497200e-02f,  1.047200e-02f,
                2.590490e-01f,  2.342060e-01f,  2.093630e-01f,  1.845200e-01f,  1.596770e-01f,  1.348340e-01f,  1.099910e-01f,  8.514800e-02f,  6.030500e-02f,  3.546200e-02f,  1.061900e-02f,
                2.626260e-01f,  2.374400e-01f,  2.122540e-01f,  1.870680e-01f,  1.618820e-01f,  1.366960e-01f,  1.115100e-01f,  8.632400e-02f,  6.113800e-02f,  3.595200e-02f,  1.076600e-02f,
                2.662030e-01f,  2.406740e-01f,  2.151450e-01f,  1.896160e-01f,  1.640870e-01f,  1.385580e-01f,  1.130290e-01f,  8.750000e-02f,  6.197100e-02f,  3.644200e-02f,  1.091300e-02f,
                2.697800e-01f,  2.439080e-01f,  2.180360e-01f,  1.921640e-01f,  1.662920e-01f,  1.404200e-01f,  1.145480e-01f,  8.867600e-02f,  6.280400e-02f,  3.693200e-02f,  1.106000e-02f,
                2.733570e-01f,  2.471420e-01f,  2.209270e-01f,  1.947120e-01f,  1.684970e-01f,  1.422820e-01f,  1.160670e-01f,  8.985200e-02f,  6.363700e-02f,  3.742200e-02f,  1.120700e-02f,
                2.769340e-01f,  2.503760e-01f,  2.238180e-01f,  1.972600e-01f,  1.707020e-01f,  1.441440e-01f,  1.175860e-01f,  9.102800e-02f,  6.447000e-02f,  3.791200e-02f,  1.135400e-02f,
                2.805110e-01f,  2.536100e-01f,  2.267090e-01f,  1.998080e-01f,  1.729070e-01f,  1.460060e-01f,  1.191050e-01f,  9.220400e-02f,  6.530300e-02f,  3.840200e-02f,  1.150100e-02f,
                2.840880e-01f,  2.568440e-01f,  2.296000e-01f,  2.023560e-01f,  1.751120e-01f,  1.478680e-01f,  1.206240e-01f,  9.338000e-02f,  6.613600e-02f,  3.889200e-02f,  1.164800e-02f,
                2.876650e-01f,  2.600780e-01f,  2.324910e-01f,  2.049040e-01f,  1.773170e-01f,  1.497300e-01f,  1.221430e-01f,  9.455600e-02f,  6.696900e-02f,  3.938200e-02f,  1.179500e-02f,
                2.912420e-01f,  2.633120e-01f,  2.353820e-01f,  2.074520e-01f,  1.795220e-01f,  1.515920e-01f,  1.236620e-01f,  9.573200e-02f,  6.780200e-02f,  3.987200e-02f,  1.194200e-02f,
                2.948190e-01f,  2.665460e-01f,  2.382730e-01f,  2.100000e-01f,  1.817270e-01f,  1.534540e-01f,  1.251810e-01f,  9.690800e-02f,  6.863500e-02f,  4.036200e-02f,  1.208900e-02f,
                2.983960e-01f,  2.697800e-01f,  2.411640e-01f,  2.125480e-01f,  1.839320e-01f,  1.553160e-01f,  1.267000e-01f,  9.808400e-02f,  6.946800e-02f,  4.085200e-02f,  1.223600e-02f,
                3.019730e-01f,  2.730140e-01f,  2.440550e-01f,  2.150960e-01f,  1.861370e-01f,  1.571780e-01f,  1.282190e-01f,  9.926000e-02f,  7.030100e-02f,  4.134200e-02f,  1.238300e-02f,
                3.055500e-01f,  2.762480e-01f,  2.469460e-01f,  2.176440e-01f,  1.883420e-01f,  1.590400e-01f,  1.297380e-01f,  1.004360e-01f,  7.113400e-02f,  4.183200e-02f,  1.253000e-02f,
                3.091270e-01f,  2.794820e-01f,  2.498370e-01f,  2.201920e-01f,  1.905470e-01f,  1.609020e-01f,  1.312570e-01f,  1.016120e-01f,  7.196700e-02f,  4.232200e-02f,  1.267700e-02f,
                3.127040e-01f,  2.827160e-01f,  2.527280e-01f,  2.227400e-01f,  1.927520e-01f,  1.627640e-01f,  1.327760e-01f,  1.027880e-01f,  7.280000e-02f,  4.281200e-02f,  1.282400e-02f,
                3.162810e-01f,  2.859500e-01f,  2.556190e-01f,  2.252880e-01f,  1.949570e-01f,  1.646260e-01f,  1.342950e-01f,  1.039640e-01f,  7.363300e-02f,  4.330200e-02f,  1.297100e-02f,
                3.198580e-01f,  2.891840e-01f,  2.585100e-01f,  2.278360e-01f,  1.971620e-01f,  1.664880e-01f,  1.358140e-01f,  1.051400e-01f,  7.446600e-02f,  4.379200e-02f,  1.311800e-02f,
                3.234350e-01f,  2.924180e-01f,  2.614010e-01f,  2.303840e-01f,  1.993670e-01f,  1.683500e-01f,  1.373330e-01f,  1.063160e-01f,  7.529900e-02f,  4.428200e-02f,  1.326500e-02f,
                3.270120e-01f,  2.956520e-01f,  2.642920e-01f,  2.329320e-01f,  2.015720e-01f,  1.702120e-01f,  1.388520e-01f,  1.074920e-01f,  7.613200e-02f,  4.477200e-02f,  1.341200e-02f,
                3.305890e-01f,  2.988860e-01f,  2.671830e-01f,  2.354800e-01f,  2.037770e-01f,  1.720740e-01f,  1.403710e-01f,  1.086680e-01f,  7.696500e-02f,  4.526200e-02f,  1.355900e-02f,
                3.341660e-01f,  3.021200e-01f,  2.700740e-01f,  2.380280e-01f,  2.059820e-01f,  1.739360e-01f,  1.418900e-01f,  1.098440e-01f,  7.779800e-02f,  4.575200e-02f,  1.370600e-02f,
                3.377430e-01f,  3.053540e-01f,  2.729650e-01f,  2.405760e-01f,  2.081870e-01f,  1.757980e-01f,  1.434090e-01f,  1.110200e-01f,  7.863100e-02f,  4.624200e-02f,  1.385300e-02f,
                3.413200e-01f,  3.085880e-01f,  2.758560e-01f,  2.431240e-01f,  2.103920e-01f,  1.776600e-01f,  1.449280e-01f,  1.121960e-01f,  7.946400e-02f,  4.673200e-02f,  1.400000e-02f,
                3.448970e-01f,  3.118220e-01f,  2.787470e-01f,  2.456720e-01f,  2.125970e-01f,  1.795220e-01f,  1.464470e-01f,  1.133720e-01f,  8.029700e-02f,  4.722200e-02f,  1.414700e-02f,
                3.484740e-01f,  3.150560e-01f,  2.816380e-01f,  2.482200e-01f,  2.148020e-01f,  1.813840e-01f,  1.479660e-01f,  1.145480e-01f,  8.113000e-02f,  4.771200e-02f,  1.429400e-02f,
                3.520510e-01f,  3.182900e-01f,  2.845290e-01f,  2.507680e-01f,  2.170070e-01f,  1.832460e-01f,  1.494850e-01f,  1.157240e-01f,  8.196300e-02f,  4.820200e-02f,  1.444100e-02f,
                3.556280e-01f,  3.215240e-01f,  2.874200e-01f,  2.533160e-01f,  2.192120e-01f,  1.851080e-01f,  1.510040e-01f,  1.169000e-01f,  8.279600e-02f,  4.869200e-02f,  1.458800e-02f,
                3.592050e-01f,  3.247580e-01f,  2.903110e-01f,  2.558640e-01f,  2.214170e-01f,  1.869700e-01f,  1.525230e-01f,  1.180760e-01f,  8.362900e-02f,  4.918200e-02f,  1.473500e-02f,
                3.627820e-01f,  3.279920e-01f,  2.932020e-01f,  2.584120e-01f,  2.236220e-01f,  1.888320e-01f,  1.540420e-01f,  1.192520e-01f,  8.446200e-02f,  4.967200e-02f,  1.488200e-02f,
                3.663590e-01f,  3.312260e-01f,  2.960930e-01f,  2.609600e-01f,  2.258270e-01f,  1.906940e-01f,  1.555610e-01f,  1.204280e-01f,  8.529500e-02f,  5.016200e-02f,  1.502900e-02f,
                3.699360e-01f,  3.344600e-01f,  2.989840e-01f,  2.635080e-01f,  2.280320e-01f,  1.925560e-01f,  1.570800e-01f,  1.216040e-01f,  8.612800e-02f,  5.065200e-02f,  1.517600e-02f,
                3.735130e-01f,  3.376940e-01f,  3.018750e-01f,  2.660560e-01f,  2.302370e-01f,  1.944180e-01f,  1.585990e-01f,  1.227800e-01f,  8.696100e-02f,  5.114200e-02f,  1.532300e-02f,
                3.770900e-01f,  3.409280e-01f,  3.047660e-01f,  2.686040e-01f,  2.324420e-01f,  1.962800e-01f,  1.601180e-01f,  1.239560e-01f,  8.779400e-02f,  5.163200e-02f,  1.547000e-02f,
                3.806670e-01f,  3.441620e-01f,  3.076570e-01f,  2.711520e-01f,  2.346470e-01f,  1.981420e-01f,  1.616370e-01f,  1.251320e-01f,  8.862700e-02f,  5.212200e-02f,  1.561700e-02f,
                3.842440e-01f,  3.473960e-01f,  3.105480e-01f,  2.737000e-01f,  2.368520e-01f,  2.000040e-01f,  1.631560e-01f,  1.263080e-01f,  8.946000e-02f,  5.261200e-02f,  1.576400e-02f,
                3.878210e-01f,  3.506300e-01f,  3.134390e-01f,  2.762480e-01f,  2.390570e-01f,  2.018660e-01f,  1.646750e-01f,  1.274840e-01f,  9.029300e-02f,  5.310200e-02f,  1.591100e-02f,
                3.913980e-01f,  3.538640e-01f,  3.163300e-01f,  2.787960e-01f,  2.412620e-01f,  2.037280e-01f,  1.661940e-01f,  1.286600e-01f,  9.112600e-02f,  5.359200e-02f,  1.605800e-02f,
                3.949750e-01f,  3.570980e-01f,  3.192210e-01f,  2.813440e-01f,  2.434670e-01f,  2.055900e-01f,  1.677130e-01f,  1.298360e-01f,  9.195900e-02f,  5.408200e-02f,  1.620500e-02f,
                3.985520e-01f,  3.603320e-01f,  3.221120e-01f,  2.838920e-01f,  2.456720e-01f,  2.074520e-01f,  1.692320e-01f,  1.310120e-01f,  9.279200e-02f,  5.457200e-02f,  1.635200e-02f,
                4.021290e-01f,  3.635660e-01f,  3.250030e-01f,  2.864400e-01f,  2.478770e-01f,  2.093140e-01f,  1.707510e-01f,  1.321880e-01f,  9.362500e-02f,  5.506200e-02f,  1.649900e-02f,
                4.057060e-01f,  3.668000e-01f,  3.278940e-01f,  2.889880e-01f,  2.500820e-01f,  2.111760e-01f,  1.722700e-01f,  1.333640e-01f,  9.445800e-02f,  5.555200e-02f,  1.664600e-02f,
                4.092830e-01f,  3.700340e-01f,  3.307850e-01f,  2.915360e-01f,  2.522870e-01f,  2.130380e-01f,  1.737890e-01f,  1.345400e-01f,  9.529100e-02f,  5.604200e-02f,  1.679300e-02f,
                4.128600e-01f,  3.732680e-01f,  3.336760e-01f,  2.940840e-01f,  2.544920e-01f,  2.149000e-01f,  1.753080e-01f,  1.357160e-01f,  9.612400e-02f,  5.653200e-02f,  1.694000e-02f,
                4.164370e-01f,  3.765020e-01f,  3.365670e-01f,  2.966320e-01f,  2.566970e-01f,  2.167620e-01f,  1.768270e-01f,  1.368920e-01f,  9.695700e-02f,  5.702200e-02f,  1.708700e-02f,
                4.200140e-01f,  3.797360e-01f,  3.394580e-01f,  2.991800e-01f,  2.589020e-01f,  2.186240e-01f,  1.783460e-01f,  1.380680e-01f,  9.779000e-02f,  5.751200e-02f,  1.723400e-02f,
                4.235910e-01f,  3.829700e-01f,  3.423490e-01f,  3.017280e-01f,  2.611070e-01f,  2.204860e-01f,  1.798650e-01f,  1.392440e-01f,  9.862300e-02f,  5.800200e-02f,  1.738100e-02f,
                4.271680e-01f,  3.862040e-01f,  3.452400e-01f,  3.042760e-01f,  2.633120e-01f,  2.223480e-01f,  1.813840e-01f,  1.404200e-01f,  9.945600e-02f,  5.849200e-02f,  1.752800e-02f,
                4.307450e-01f,  3.894380e-01f,  3.481310e-01f,  3.068240e-01f,  2.655170e-01f,  2.242100e-01f,  1.829030e-01f,  1.415960e-01f,  1.002890e-01f,  5.898200e-02f,  1.767500e-02f,
                4.343220e-01f,  3.926720e-01f,  3.510220e-01f,  3.093720e-01f,  2.677220e-01f,  2.260720e-01f,  1.844220e-01f,  1.427720e-01f,  1.011220e-01f,  5.947200e-02f,  1.782200e-02f,
                4.378990e-01f,  3.959060e-01f,  3.539130e-01f,  3.119200e-01f,  2.699270e-01f,  2.279340e-01f,  1.859410e-01f,  1.439480e-01f,  1.019550e-01f,  5.996200e-02f,  1.796900e-02f,
                4.414760e-01f,  3.991400e-01f,  3.568040e-01f,  3.144680e-01f,  2.721320e-01f,  2.297960e-01f,  1.874600e-01f,  1.451240e-01f,  1.027880e-01f,  6.045200e-02f,  1.811600e-02f,
                4.450530e-01f,  4.023740e-01f,  3.596950e-01f,  3.170160e-01f,  2.743370e-01f,  2.316580e-01f,  1.889790e-01f,  1.463000e-01f,  1.036210e-01f,  6.094200e-02f,  1.826300e-02f,
                4.486300e-01f,  4.056080e-01f,  3.625860e-01f,  3.195640e-01f,  2.765420e-01f,  2.335200e-01f,  1.904980e-01f,  1.474760e-01f,  1.044540e-01f,  6.143200e-02f,  1.841000e-02f,
                4.522070e-01f,  4.088420e-01f,  3.654770e-01f,  3.221120e-01f,  2.787470e-01f,  2.353820e-01f,  1.920170e-01f,  1.486520e-01f,  1.052870e-01f,  6.192200e-02f,  1.855700e-02f,
                4.557840e-01f,  4.120760e-01f,  3.683680e-01f,  3.246600e-01f,  2.809520e-01f,  2.372440e-01f,  1.935360e-01f,  1.498280e-01f,  1.061200e-01f,  6.241200e-02f,  1.870400e-02f,
                4.593610e-01f,  4.153100e-01f,  3.712590e-01f,  3.272080e-01f,  2.831570e-01f,  2.391060e-01f,  1.950550e-01f,  1.510040e-01f,  1.069530e-01f,  6.290200e-02f,  1.885100e-02f,
                4.629380e-01f,  4.185440e-01f,  3.741500e-01f,  3.297560e-01f,  2.853620e-01f,  2.409680e-01f,  1.965740e-01f,  1.521800e-01f,  1.077860e-01f,  6.339200e-02f,  1.899800e-02f,
                4.665150e-01f,  4.217780e-01f,  3.770410e-01f,  3.323040e-01f,  2.875670e-01f,  2.428300e-01f,  1.980930e-01f,  1.533560e-01f,  1.086190e-01f,  6.388200e-02f,  1.914500e-02f,
                4.700920e-01f,  4.250120e-01f,  3.799320e-01f,  3.348520e-01f,  2.897720e-01f,  2.446920e-01f,  1.996120e-01f,  1.545320e-01f,  1.094520e-01f,  6.437200e-02f,  1.929200e-02f,
                4.736690e-01f,  4.282460e-01f,  3.828230e-01f,  3.374000e-01f,  2.919770e-01f,  2.465540e-01f,  2.011310e-01f,  1.557080e-01f,  1.102850e-01f,  6.486200e-02f,  1.943900e-02f,
                4.772460e-01f,  4.314800e-01f,  3.857140e-01f,  3.399480e-01f,  2.941820e-01f,  2.484160e-01f,  2.026500e-01f,  1.568840e-01f,  1.111180e-01f,  6.535200e-02f,  1.958600e-02f,
                4.808230e-01f,  4.347140e-01f,  3.886050e-01f,  3.424960e-01f,  2.963870e-01f,  2.502780e-01f,  2.041690e-01f,  1.580600e-01f,  1.119510e-01f,  6.584200e-02f,  1.973300e-02f,
                4.844000e-01f,  4.379480e-01f,  3.914960e-01f,  3.450440e-01f,  2.985920e-01f,  2.521400e-01f,  2.056880e-01f,  1.592360e-01f,  1.127840e-01f,  6.633200e-02f,  1.988000e-02f,
                4.879770e-01f,  4.411820e-01f,  3.943870e-01f,  3.475920e-01f,  3.007970e-01f,  2.540020e-01f,  2.072070e-01f,  1.604120e-01f,  1.136170e-01f,  6.682200e-02f,  2.002700e-02f,
                4.915540e-01f,  4.444160e-01f,  3.972780e-01f,  3.501400e-01f,  3.030020e-01f,  2.558640e-01f,  2.087260e-01f,  1.615880e-01f,  1.144500e-01f,  6.731200e-02f,  2.017400e-02f,
                4.951310e-01f,  4.476500e-01f,  4.001690e-01f,  3.526880e-01f,  3.052070e-01f,  2.577260e-01f,  2.102450e-01f,  1.627640e-01f,  1.152830e-01f,  6.780200e-02f,  2.032100e-02f,
                4.987080e-01f,  4.508840e-01f,  4.030600e-01f,  3.552360e-01f,  3.074120e-01f,  2.595880e-01f,  2.117640e-01f,  1.639400e-01f,  1.161160e-01f,  6.829200e-02f,  2.046800e-02f,
                5.022850e-01f,  4.541180e-01f,  4.059510e-01f,  3.577840e-01f,  3.096170e-01f,  2.614500e-01f,  2.132830e-01f,  1.651160e-01f,  1.169490e-01f,  6.878200e-02f,  2.061500e-02f,
                5.058620e-01f,  4.573520e-01f,  4.088420e-01f,  3.603320e-01f,  3.118220e-01f,  2.633120e-01f,  2.148020e-01f,  1.662920e-01f,  1.177820e-01f,  6.927200e-02f,  2.076200e-02f,
                5.094390e-01f,  4.605860e-01f,  4.117330e-01f,  3.628800e-01f,  3.140270e-01f,  2.651740e-01f,  2.163210e-01f,  1.674680e-01f,  1.186150e-01f,  6.976200e-02f,  2.090900e-02f,
                5.130160e-01f,  4.638200e-01f,  4.146240e-01f,  3.654280e-01f,  3.162320e-01f,  2.670360e-01f,  2.178400e-01f,  1.686440e-01f,  1.194480e-01f,  7.025200e-02f,  2.105600e-02f,
                5.165930e-01f,  4.670540e-01f,  4.175150e-01f,  3.679760e-01f,  3.184370e-01f,  2.688980e-01f,  2.193590e-01f,  1.698200e-01f,  1.202810e-01f,  7.074200e-02f,  2.120300e-02f,
                5.201700e-01f,  4.702880e-01f,  4.204060e-01f,  3.705240e-01f,  3.206420e-01f,  2.707600e-01f,  2.208780e-01f,  1.709960e-01f,  1.211140e-01f,  7.123200e-02f,  2.135000e-02f,
                5.237470e-01f,  4.735220e-01f,  4.232970e-01f,  3.730720e-01f,  3.228470e-01f,  2.726220e-01f,  2.223970e-01f,  1.721720e-01f,  1.219470e-01f,  7.172200e-02f,  2.149700e-02f,
                5.273240e-01f,  4.767560e-01f,  4.261880e-01f,  3.756200e-01f,  3.250520e-01f,  2.744840e-01f,  2.239160e-01f,  1.733480e-01f,  1.227800e-01f,  7.221200e-02f,  2.164400e-02f,
                5.309010e-01f,  4.799900e-01f,  4.290790e-01f,  3.781680e-01f,  3.272570e-01f,  2.763460e-01f,  2.254350e-01f,  1.745240e-01f,  1.236130e-01f,  7.270200e-02f,  2.179100e-02f,
                5.344780e-01f,  4.832240e-01f,  4.319700e-01f,  3.807160e-01f,  3.294620e-01f,  2.782080e-01f,  2.269540e-01f,  1.757000e-01f,  1.244460e-01f,  7.319200e-02f,  2.193800e-02f,
                5.380550e-01f,  4.864580e-01f,  4.348610e-01f,  3.832640e-01f,  3.316670e-01f,  2.800700e-01f,  2.284730e-01f,  1.768760e-01f,  1.252790e-01f,  7.368200e-02f,  2.208500e-02f,
                5.416320e-01f,  4.896920e-01f,  4.377520e-01f,  3.858120e-01f,  3.338720e-01f,  2.819320e-01f,  2.299920e-01f,  1.780520e-01f,  1.261120e-01f,  7.417200e-02f,  2.223200e-02f,
                5.452090e-01f,  4.929260e-01f,  4.406430e-01f,  3.883600e-01f,  3.360770e-01f,  2.837940e-01f,  2.315110e-01f,  1.792280e-01f,  1.269450e-01f,  7.466200e-02f,  2.237900e-02f,
                5.487860e-01f,  4.961600e-01f,  4.435340e-01f,  3.909080e-01f,  3.382820e-01f,  2.856560e-01f,  2.330300e-01f,  1.804040e-01f,  1.277780e-01f,  7.515200e-02f,  2.252600e-02f,
                5.523630e-01f,  4.993940e-01f,  4.464250e-01f,  3.934560e-01f,  3.404870e-01f,  2.875180e-01f,  2.345490e-01f,  1.815800e-01f,  1.286110e-01f,  7.564200e-02f,  2.267300e-02f,
                5.559400e-01f,  5.026280e-01f,  4.493160e-01f,  3.960040e-01f,  3.426920e-01f,  2.893800e-01f,  2.360680e-01f,  1.827560e-01f,  1.294440e-01f,  7.613200e-02f,  2.282000e-02f,
                5.595170e-01f,  5.058620e-01f,  4.522070e-01f,  3.985520e-01f,  3.448970e-01f,  2.912420e-01f,  2.375870e-01f,  1.839320e-01f,  1.302770e-01f,  7.662200e-02f,  2.296700e-02f,
                5.630940e-01f,  5.090960e-01f,  4.550980e-01f,  4.011000e-01f,  3.471020e-01f,  2.931040e-01f,  2.391060e-01f,  1.851080e-01f,  1.311100e-01f,  7.711200e-02f,  2.311400e-02f,
                5.666710e-01f,  5.123300e-01f,  4.579890e-01f,  4.036480e-01f,  3.493070e-01f,  2.949660e-01f,  2.406250e-01f,  1.862840e-01f,  1.319430e-01f,  7.760200e-02f,  2.326100e-02f,
                5.702480e-01f,  5.155640e-01f,  4.608800e-01f,  4.061960e-01f,  3.515120e-01f,  2.968280e-01f,  2.421440e-01f,  1.874600e-01f,  1.327760e-01f,  7.809200e-02f,  2.340800e-02f,
                5.738250e-01f,  5.187980e-01f,  4.637710e-01f,  4.087440e-01f,  3.537170e-01f,  2.986900e-01f,  2.436630e-01f,  1.886360e-01f,  1.336090e-01f,  7.858200e-02f,  2.355500e-02f,
                5.774020e-01f,  5.220320e-01f,  4.666620e-01f,  4.112920e-01f,  3.559220e-01f,  3.005520e-01f,  2.451820e-01f,  1.898120e-01f,  1.344420e-01f,  7.907200e-02f,  2.370200e-02f,
                5.809790e-01f,  5.252660e-01f,  4.695530e-01f,  4.138400e-01f,  3.581270e-01f,  3.024140e-01f,  2.467010e-01f,  1.909880e-01f,  1.352750e-01f,  7.956200e-02f,  2.384900e-02f,
                5.845560e-01f,  5.285000e-01f,  4.724440e-01f,  4.163880e-01f,  3.603320e-01f,  3.042760e-01f,  2.482200e-01f,  1.921640e-01f,  1.361080e-01f,  8.005200e-02f,  2.399600e-02f,
                5.881330e-01f,  5.317340e-01f,  4.753350e-01f,  4.189360e-01f,  3.625370e-01f,  3.061380e-01f,  2.497390e-01f,  1.933400e-01f,  1.369410e-01f,  8.054200e-02f,  2.414300e-02f,
                5.917100e-01f,  5.349680e-01f,  4.782260e-01f,  4.214840e-01f,  3.647420e-01f,  3.080000e-01f,  2.512580e-01f,  1.945160e-01f,  1.377740e-01f,  8.103200e-02f,  2.429000e-02f,
                5.952870e-01f,  5.382020e-01f,  4.811170e-01f,  4.240320e-01f,  3.669470e-01f,  3.098620e-01f,  2.527770e-01f,  1.956920e-01f,  1.386070e-01f,  8.152200e-02f,  2.443700e-02f,
                5.988640e-01f,  5.414360e-01f,  4.840080e-01f,  4.265800e-01f,  3.691520e-01f,  3.117240e-01f,  2.542960e-01f,  1.968680e-01f,  1.394400e-01f,  8.201200e-02f,  2.458400e-02f,
                6.024410e-01f,  5.446700e-01f,  4.868990e-01f,  4.291280e-01f,  3.713570e-01f,  3.135860e-01f,  2.558150e-01f,  1.980440e-01f,  1.402730e-01f,  8.250200e-02f,  2.473100e-02f,
                6.060180e-01f,  5.479040e-01f,  4.897900e-01f,  4.316760e-01f,  3.735620e-01f,  3.154480e-01f,  2.573340e-01f,  1.992200e-01f,  1.411060e-01f,  8.299200e-02f,  2.487800e-02f,
                6.095950e-01f,  5.511380e-01f,  4.926810e-01f,  4.342240e-01f,  3.757670e-01f,  3.173100e-01f,  2.588530e-01f,  2.003960e-01f,  1.419390e-01f,  8.348200e-02f,  2.502500e-02f,
                6.131720e-01f,  5.543720e-01f,  4.955720e-01f,  4.367720e-01f,  3.779720e-01f,  3.191720e-01f,  2.603720e-01f,  2.015720e-01f,  1.427720e-01f,  8.397200e-02f,  2.517200e-02f,
                6.167490e-01f,  5.576060e-01f,  4.984630e-01f,  4.393200e-01f,  3.801770e-01f,  3.210340e-01f,  2.618910e-01f,  2.027480e-01f,  1.436050e-01f,  8.446200e-02f,  2.531900e-02f,
                6.203260e-01f,  5.608400e-01f,  5.013540e-01f,  4.418680e-01f,  3.823820e-01f,  3.228960e-01f,  2.634100e-01f,  2.039240e-01f,  1.444380e-01f,  8.495200e-02f,  2.546600e-02f,
                6.239030e-01f,  5.640740e-01f,  5.042450e-01f,  4.444160e-01f,  3.845870e-01f,  3.247580e-01f,  2.649290e-01f,  2.051000e-01f,  1.452710e-01f,  8.544200e-02f,  2.561300e-02f,
                6.274800e-01f,  5.673080e-01f,  5.071360e-01f,  4.469640e-01f,  3.867920e-01f,  3.266200e-01f,  2.664480e-01f,  2.062760e-01f,  1.461040e-01f,  8.593200e-02f,  2.576000e-02f,
                6.310570e-01f,  5.705420e-01f,  5.100270e-01f,  4.495120e-01f,  3.889970e-01f,  3.284820e-01f,  2.679670e-01f,  2.074520e-01f,  1.469370e-01f,  8.642200e-02f,  2.590700e-02f,
                6.346340e-01f,  5.737760e-01f,  5.129180e-01f,  4.520600e-01f,  3.912020e-01f,  3.303440e-01f,  2.694860e-01f,  2.086280e-01f,  1.477700e-01f,  8.691200e-02f,  2.605400e-02f,
                6.382110e-01f,  5.770100e-01f,  5.158090e-01f,  4.546080e-01f,  3.934070e-01f,  3.322060e-01f,  2.710050e-01f,  2.098040e-01f,  1.486030e-01f,  8.740200e-02f,  2.620100e-02f,
                6.417880e-01f,  5.802440e-01f,  5.187000e-01f,  4.571560e-01f,  3.956120e-01f,  3.340680e-01f,  2.725240e-01f,  2.109800e-01f,  1.494360e-01f,  8.789200e-02f,  2.634800e-02f,
                6.453650e-01f,  5.834780e-01f,  5.215910e-01f,  4.597040e-01f,  3.978170e-01f,  3.359300e-01f,  2.740430e-01f,  2.121560e-01f,  1.502690e-01f,  8.838200e-02f,  2.649500e-02f,
                6.489420e-01f,  5.867120e-01f,  5.244820e-01f,  4.622520e-01f,  4.000220e-01f,  3.377920e-01f,  2.755620e-01f,  2.133320e-01f,  1.511020e-01f,  8.887200e-02f,  2.664200e-02f,
                6.525190e-01f,  5.899460e-01f,  5.273730e-01f,  4.648000e-01f,  4.022270e-01f,  3.396540e-01f,  2.770810e-01f,  2.145080e-01f,  1.519350e-01f,  8.936200e-02f,  2.678900e-02f,
                6.560960e-01f,  5.931800e-01f,  5.302640e-01f,  4.673480e-01f,  4.044320e-01f,  3.415160e-01f,  2.786000e-01f,  2.156840e-01f,  1.527680e-01f,  8.985200e-02f,  2.693600e-02f,
                6.596730e-01f,  5.964140e-01f,  5.331550e-01f,  4.698960e-01f,  4.066370e-01f,  3.433780e-01f,  2.801190e-01f,  2.168600e-01f,  1.536010e-01f,  9.034200e-02f,  2.708300e-02f,
                6.632500e-01f,  5.996480e-01f,  5.360460e-01f,  4.724440e-01f,  4.088420e-01f,  3.452400e-01f,  2.816380e-01f,  2.180360e-01f,  1.544340e-01f,  9.083200e-02f,  2.723000e-02f,
                6.668270e-01f,  6.028820e-01f,  5.389370e-01f,  4.749920e-01f,  4.110470e-01f,  3.471020e-01f,  2.831570e-01f,  2.192120e-01f,  1.552670e-01f,  9.132200e-02f,  2.737700e-02f,
                6.704040e-01f,  6.061160e-01f,  5.418280e-01f,  4.775400e-01f,  4.132520e-01f,  3.489640e-01f,  2.846760e-01f,  2.203880e-01f,  1.561000e-01f,  9.181200e-02f,  2.752400e-02f,
                6.739810e-01f,  6.093500e-01f,  5.447190e-01f,  4.800880e-01f,  4.154570e-01f,  3.508260e-01f,  2.861950e-01f,  2.215640e-01f,  1.569330e-01f,  9.230200e-02f,  2.767100e-02f,
                6.775580e-01f,  6.125840e-01f,  5.476100e-01f,  4.826360e-01f,  4.176620e-01f,  3.526880e-01f,  2.877140e-01f,  2.227400e-01f,  1.577660e-01f,  9.279200e-02f,  2.781800e-02f,
                6.811350e-01f,  6.158180e-01f,  5.505010e-01f,  4.851840e-01f,  4.198670e-01f,  3.545500e-01f,  2.892330e-01f,  2.239160e-01f,  1.585990e-01f,  9.328200e-02f,  2.796500e-02f,
                6.847120e-01f,  6.190520e-01f,  5.533920e-01f,  4.877320e-01f,  4.220720e-01f,  3.564120e-01f,  2.907520e-01f,  2.250920e-01f,  1.594320e-01f,  9.377200e-02f,  2.811200e-02f,
                6.882890e-01f,  6.222860e-01f,  5.562830e-01f,  4.902800e-01f,  4.242770e-01f,  3.582740e-01f,  2.922710e-01f,  2.262680e-01f,  1.602650e-01f,  9.426200e-02f,  2.825900e-02f,
                6.918660e-01f,  6.255200e-01f,  5.591740e-01f,  4.928280e-01f,  4.264820e-01f,  3.601360e-01f,  2.937900e-01f,  2.274440e-01f,  1.610980e-01f,  9.475200e-02f,  2.840600e-02f,
                6.954430e-01f,  6.287540e-01f,  5.620650e-01f,  4.953760e-01f,  4.286870e-01f,  3.619980e-01f,  2.953090e-01f,  2.286200e-01f,  1.619310e-01f,  9.524200e-02f,  2.855300e-02f,
                6.990200e-01f,  6.319880e-01f,  5.649560e-01f,  4.979240e-01f,  4.308920e-01f,  3.638600e-01f,  2.968280e-01f,  2.297960e-01f,  1.627640e-01f,  9.573200e-02f,  2.870000e-02f,
                7.025970e-01f,  6.352220e-01f,  5.678470e-01f,  5.004720e-01f,  4.330970e-01f,  3.657220e-01f,  2.983470e-01f,  2.309720e-01f,  1.635970e-01f,  9.622200e-02f,  2.884700e-02f,
                7.061740e-01f,  6.384560e-01f,  5.707380e-01f,  5.030200e-01f,  4.353020e-01f,  3.675840e-01f,  2.998660e-01f,  2.321480e-01f,  1.644300e-01f,  9.671200e-02f,  2.899400e-02f,
                7.097510e-01f,  6.416900e-01f,  5.736290e-01f,  5.055680e-01f,  4.375070e-01f,  3.694460e-01f,  3.013850e-01f,  2.333240e-01f,  1.652630e-01f,  9.720200e-02f,  2.914100e-02f,
                7.133280e-01f,  6.449240e-01f,  5.765200e-01f,  5.081160e-01f,  4.397120e-01f,  3.713080e-01f,  3.029040e-01f,  2.345000e-01f,  1.660960e-01f,  9.769200e-02f,  2.928800e-02f,
                7.169050e-01f,  6.481580e-01f,  5.794110e-01f,  5.106640e-01f,  4.419170e-01f,  3.731700e-01f,  3.044230e-01f,  2.356760e-01f,  1.669290e-01f,  9.818200e-02f,  2.943500e-02f,
                7.204820e-01f,  6.513920e-01f,  5.823020e-01f,  5.132120e-01f,  4.441220e-01f,  3.750320e-01f,  3.059420e-01f,  2.368520e-01f,  1.677620e-01f,  9.867200e-02f,  2.958200e-02f,
                7.240590e-01f,  6.546260e-01f,  5.851930e-01f,  5.157600e-01f,  4.463270e-01f,  3.768940e-01f,  3.074610e-01f,  2.380280e-01f,  1.685950e-01f,  9.916200e-02f,  2.972900e-02f,
                7.276360e-01f,  6.578600e-01f,  5.880840e-01f,  5.183080e-01f,  4.485320e-01f,  3.787560e-01f,  3.089800e-01f,  2.392040e-01f,  1.694280e-01f,  9.965200e-02f,  2.987600e-02f,
                7.312130e-01f,  6.610940e-01f,  5.909750e-01f,  5.208560e-01f,  4.507370e-01f,  3.806180e-01f,  3.104990e-01f,  2.403800e-01f,  1.702610e-01f,  1.001420e-01f,  3.002300e-02f,
                7.347900e-01f,  6.643280e-01f,  5.938660e-01f,  5.234040e-01f,  4.529420e-01f,  3.824800e-01f,  3.120180e-01f,  2.415560e-01f,  1.710940e-01f,  1.006320e-01f,  3.017000e-02f,
                7.383670e-01f,  6.675620e-01f,  5.967570e-01f,  5.259520e-01f,  4.551470e-01f,  3.843420e-01f,  3.135370e-01f,  2.427320e-01f,  1.719270e-01f,  1.011220e-01f,  3.031700e-02f,
                7.419440e-01f,  6.707960e-01f,  5.996480e-01f,  5.285000e-01f,  4.573520e-01f,  3.862040e-01f,  3.150560e-01f,  2.439080e-01f,  1.727600e-01f,  1.016120e-01f,  3.046400e-02f,
                7.455210e-01f,  6.740300e-01f,  6.025390e-01f,  5.310480e-01f,  4.595570e-01f,  3.880660e-01f,  3.165750e-01f,  2.450840e-01f,  1.735930e-01f,  1.021020e-01f,  3.061100e-02f,
                7.490980e-01f,  6.772640e-01f,  6.054300e-01f,  5.335960e-01f,  4.617620e-01f,  3.899280e-01f,  3.180940e-01f,  2.462600e-01f,  1.744260e-01f,  1.025920e-01f,  3.075800e-02f,
                7.526750e-01f,  6.804980e-01f,  6.083210e-01f,  5.361440e-01f,  4.639670e-01f,  3.917900e-01f,  3.196130e-01f,  2.474360e-01f,  1.752590e-01f,  1.030820e-01f,  3.090500e-02f,
                7.562520e-01f,  6.837320e-01f,  6.112120e-01f,  5.386920e-01f,  4.661720e-01f,  3.936520e-01f,  3.211320e-01f,  2.486120e-01f,  1.760920e-01f,  1.035720e-01f,  3.105200e-02f,
                7.598290e-01f,  6.869660e-01f,  6.141030e-01f,  5.412400e-01f,  4.683770e-01f,  3.955140e-01f,  3.226510e-01f,  2.497880e-01f,  1.769250e-01f,  1.040620e-01f,  3.119900e-02f,
                7.634060e-01f,  6.902000e-01f,  6.169940e-01f,  5.437880e-01f,  4.705820e-01f,  3.973760e-01f,  3.241700e-01f,  2.509640e-01f,  1.777580e-01f,  1.045520e-01f,  3.134600e-02f,
                7.669830e-01f,  6.934340e-01f,  6.198850e-01f,  5.463360e-01f,  4.727870e-01f,  3.992380e-01f,  3.256890e-01f,  2.521400e-01f,  1.785910e-01f,  1.050420e-01f,  3.149300e-02f,
                7.705600e-01f,  6.966680e-01f,  6.227760e-01f,  5.488840e-01f,  4.749920e-01f,  4.011000e-01f,  3.272080e-01f,  2.533160e-01f,  1.794240e-01f,  1.055320e-01f,  3.164000e-02f,
                7.741370e-01f,  6.999020e-01f,  6.256670e-01f,  5.514320e-01f,  4.771970e-01f,  4.029620e-01f,  3.287270e-01f,  2.544920e-01f,  1.802570e-01f,  1.060220e-01f,  3.178700e-02f,
                7.777140e-01f,  7.031360e-01f,  6.285580e-01f,  5.539800e-01f,  4.794020e-01f,  4.048240e-01f,  3.302460e-01f,  2.556680e-01f,  1.810900e-01f,  1.065120e-01f,  3.193400e-02f,
                7.812910e-01f,  7.063700e-01f,  6.314490e-01f,  5.565280e-01f,  4.816070e-01f,  4.066860e-01f,  3.317650e-01f,  2.568440e-01f,  1.819230e-01f,  1.070020e-01f,  3.208100e-02f,
                7.848680e-01f,  7.096040e-01f,  6.343400e-01f,  5.590760e-01f,  4.838120e-01f,  4.085480e-01f,  3.332840e-01f,  2.580200e-01f,  1.827560e-01f,  1.074920e-01f,  3.222800e-02f,
                7.884450e-01f,  7.128380e-01f,  6.372310e-01f,  5.616240e-01f,  4.860170e-01f,  4.104100e-01f,  3.348030e-01f,  2.591960e-01f,  1.835890e-01f,  1.079820e-01f,  3.237500e-02f,
                7.920220e-01f,  7.160720e-01f,  6.401220e-01f,  5.641720e-01f,  4.882220e-01f,  4.122720e-01f,  3.363220e-01f,  2.603720e-01f,  1.844220e-01f,  1.084720e-01f,  3.252200e-02f,
                7.955990e-01f,  7.193060e-01f,  6.430130e-01f,  5.667200e-01f,  4.904270e-01f,  4.141340e-01f,  3.378410e-01f,  2.615480e-01f,  1.852550e-01f,  1.089620e-01f,  3.266900e-02f,
                7.991760e-01f,  7.225400e-01f,  6.459040e-01f,  5.692680e-01f,  4.926320e-01f,  4.159960e-01f,  3.393600e-01f,  2.627240e-01f,  1.860880e-01f,  1.094520e-01f,  3.281600e-02f,
                8.027530e-01f,  7.257740e-01f,  6.487950e-01f,  5.718160e-01f,  4.948370e-01f,  4.178580e-01f,  3.408790e-01f,  2.639000e-01f,  1.869210e-01f,  1.099420e-01f,  3.296300e-02f,
                8.063300e-01f,  7.290080e-01f,  6.516860e-01f,  5.743640e-01f,  4.970420e-01f,  4.197200e-01f,  3.423980e-01f,  2.650760e-01f,  1.877540e-01f,  1.104320e-01f,  3.311000e-02f,
                8.099070e-01f,  7.322420e-01f,  6.545770e-01f,  5.769120e-01f,  4.992470e-01f,  4.215820e-01f,  3.439170e-01f,  2.662520e-01f,  1.885870e-01f,  1.109220e-01f,  3.325700e-02f,
                8.134840e-01f,  7.354760e-01f,  6.574680e-01f,  5.794600e-01f,  5.014520e-01f,  4.234440e-01f,  3.454360e-01f,  2.674280e-01f,  1.894200e-01f,  1.114120e-01f,  3.340400e-02f,
                8.170610e-01f,  7.387100e-01f,  6.603590e-01f,  5.820080e-01f,  5.036570e-01f,  4.253060e-01f,  3.469550e-01f,  2.686040e-01f,  1.902530e-01f,  1.119020e-01f,  3.355100e-02f,
                8.206380e-01f,  7.419440e-01f,  6.632500e-01f,  5.845560e-01f,  5.058620e-01f,  4.271680e-01f,  3.484740e-01f,  2.697800e-01f,  1.910860e-01f,  1.123920e-01f,  3.369800e-02f,
                8.242150e-01f,  7.451780e-01f,  6.661410e-01f,  5.871040e-01f,  5.080670e-01f,  4.290300e-01f,  3.499930e-01f,  2.709560e-01f,  1.919190e-01f,  1.128820e-01f,  3.384500e-02f,
                8.277920e-01f,  7.484120e-01f,  6.690320e-01f,  5.896520e-01f,  5.102720e-01f,  4.308920e-01f,  3.515120e-01f,  2.721320e-01f,  1.927520e-01f,  1.133720e-01f,  3.399200e-02f,
                8.313690e-01f,  7.516460e-01f,  6.719230e-01f,  5.922000e-01f,  5.124770e-01f,  4.327540e-01f,  3.530310e-01f,  2.733080e-01f,  1.935850e-01f,  1.138620e-01f,  3.413900e-02f,
                8.349460e-01f,  7.548800e-01f,  6.748140e-01f,  5.947480e-01f,  5.146820e-01f,  4.346160e-01f,  3.545500e-01f,  2.744840e-01f,  1.944180e-01f,  1.143520e-01f,  3.428600e-02f,
                8.385230e-01f,  7.581140e-01f,  6.777050e-01f,  5.972960e-01f,  5.168870e-01f,  4.364780e-01f,  3.560690e-01f,  2.756600e-01f,  1.952510e-01f,  1.148420e-01f,  3.443300e-02f,
                8.421000e-01f,  7.613480e-01f,  6.805960e-01f,  5.998440e-01f,  5.190920e-01f,  4.383400e-01f,  3.575880e-01f,  2.768360e-01f,  1.960840e-01f,  1.153320e-01f,  3.458000e-02f,
                8.456770e-01f,  7.645820e-01f,  6.834870e-01f,  6.023920e-01f,  5.212970e-01f,  4.402020e-01f,  3.591070e-01f,  2.780120e-01f,  1.969170e-01f,  1.158220e-01f,  3.472700e-02f,
                8.492540e-01f,  7.678160e-01f,  6.863780e-01f,  6.049400e-01f,  5.235020e-01f,  4.420640e-01f,  3.606260e-01f,  2.791880e-01f,  1.977500e-01f,  1.163120e-01f,  3.487400e-02f,
                8.528310e-01f,  7.710500e-01f,  6.892690e-01f,  6.074880e-01f,  5.257070e-01f,  4.439260e-01f,  3.621450e-01f,  2.803640e-01f,  1.985830e-01f,  1.168020e-01f,  3.502100e-02f,
                8.564080e-01f,  7.742840e-01f,  6.921600e-01f,  6.100360e-01f,  5.279120e-01f,  4.457880e-01f,  3.636640e-01f,  2.815400e-01f,  1.994160e-01f,  1.172920e-01f,  3.516800e-02f,
                8.599850e-01f,  7.775180e-01f,  6.950510e-01f,  6.125840e-01f,  5.301170e-01f,  4.476500e-01f,  3.651830e-01f,  2.827160e-01f,  2.002490e-01f,  1.177820e-01f,  3.531500e-02f,
                8.635620e-01f,  7.807520e-01f,  6.979420e-01f,  6.151320e-01f,  5.323220e-01f,  4.495120e-01f,  3.667020e-01f,  2.838920e-01f,  2.010820e-01f,  1.182720e-01f,  3.546200e-02f,
                8.671390e-01f,  7.839860e-01f,  7.008330e-01f,  6.176800e-01f,  5.345270e-01f,  4.513740e-01f,  3.682210e-01f,  2.850680e-01f,  2.019150e-01f,  1.187620e-01f,  3.560900e-02f,
                8.707160e-01f,  7.872200e-01f,  7.037240e-01f,  6.202280e-01f,  5.367320e-01f,  4.532360e-01f,  3.697400e-01f,  2.862440e-01f,  2.027480e-01f,  1.192520e-01f,  3.575600e-02f,
                8.742930e-01f,  7.904540e-01f,  7.066150e-01f,  6.227760e-01f,  5.389370e-01f,  4.550980e-01f,  3.712590e-01f,  2.874200e-01f,  2.035810e-01f,  1.197420e-01f,  3.590300e-02f,
                8.778700e-01f,  7.936880e-01f,  7.095060e-01f,  6.253240e-01f,  5.411420e-01f,  4.569600e-01f,  3.727780e-01f,  2.885960e-01f,  2.044140e-01f,  1.202320e-01f,  3.605000e-02f,
                8.814470e-01f,  7.969220e-01f,  7.123970e-01f,  6.278720e-01f,  5.433470e-01f,  4.588220e-01f,  3.742970e-01f,  2.897720e-01f,  2.052470e-01f,  1.207220e-01f,  3.619700e-02f,
                8.850240e-01f,  8.001560e-01f,  7.152880e-01f,  6.304200e-01f,  5.455520e-01f,  4.606840e-01f,  3.758160e-01f,  2.909480e-01f,  2.060800e-01f,  1.212120e-01f,  3.634400e-02f,
                8.886010e-01f,  8.033900e-01f,  7.181790e-01f,  6.329680e-01f,  5.477570e-01f,  4.625460e-01f,  3.773350e-01f,  2.921240e-01f,  2.069130e-01f,  1.217020e-01f,  3.649100e-02f,
                8.921780e-01f,  8.066240e-01f,  7.210700e-01f,  6.355160e-01f,  5.499620e-01f,  4.644080e-01f,  3.788540e-01f,  2.933000e-01f,  2.077460e-01f,  1.221920e-01f,  3.663800e-02f,
                8.957550e-01f,  8.098580e-01f,  7.239610e-01f,  6.380640e-01f,  5.521670e-01f,  4.662700e-01f,  3.803730e-01f,  2.944760e-01f,  2.085790e-01f,  1.226820e-01f,  3.678500e-02f,
                8.993320e-01f,  8.130920e-01f,  7.268520e-01f,  6.406120e-01f,  5.543720e-01f,  4.681320e-01f,  3.818920e-01f,  2.956520e-01f,  2.094120e-01f,  1.231720e-01f,  3.693200e-02f,
                9.029090e-01f,  8.163260e-01f,  7.297430e-01f,  6.431600e-01f,  5.565770e-01f,  4.699940e-01f,  3.834110e-01f,  2.968280e-01f,  2.102450e-01f,  1.236620e-01f,  3.707900e-02f,
                9.064860e-01f,  8.195600e-01f,  7.326340e-01f,  6.457080e-01f,  5.587820e-01f,  4.718560e-01f,  3.849300e-01f,  2.980040e-01f,  2.110780e-01f,  1.241520e-01f,  3.722600e-02f,
                9.100630e-01f,  8.227940e-01f,  7.355250e-01f,  6.482560e-01f,  5.609870e-01f,  4.737180e-01f,  3.864490e-01f,  2.991800e-01f,  2.119110e-01f,  1.246420e-01f,  3.737300e-02f,
                9.136400e-01f,  8.260280e-01f,  7.384160e-01f,  6.508040e-01f,  5.631920e-01f,  4.755800e-01f,  3.879680e-01f,  3.003560e-01f,  2.127440e-01f,  1.251320e-01f,  3.752000e-02f,
                9.172170e-01f,  8.292620e-01f,  7.413070e-01f,  6.533520e-01f,  5.653970e-01f,  4.774420e-01f,  3.894870e-01f,  3.015320e-01f,  2.135770e-01f,  1.256220e-01f,  3.766700e-02f,
                9.207940e-01f,  8.324960e-01f,  7.441980e-01f,  6.559000e-01f,  5.676020e-01f,  4.793040e-01f,  3.910060e-01f,  3.027080e-01f,  2.144100e-01f,  1.261120e-01f,  3.781400e-02f,
                9.243710e-01f,  8.357300e-01f,  7.470890e-01f,  6.584480e-01f,  5.698070e-01f,  4.811660e-01f,  3.925250e-01f,  3.038840e-01f,  2.152430e-01f,  1.266020e-01f,  3.796100e-02f,
                9.279480e-01f,  8.389640e-01f,  7.499800e-01f,  6.609960e-01f,  5.720120e-01f,  4.830280e-01f,  3.940440e-01f,  3.050600e-01f,  2.160760e-01f,  1.270920e-01f,  3.810800e-02f,
                9.315250e-01f,  8.421980e-01f,  7.528710e-01f,  6.635440e-01f,  5.742170e-01f,  4.848900e-01f,  3.955630e-01f,  3.062360e-01f,  2.169090e-01f,  1.275820e-01f,  3.825500e-02f,
                9.351020e-01f,  8.454320e-01f,  7.557620e-01f,  6.660920e-01f,  5.764220e-01f,  4.867520e-01f,  3.970820e-01f,  3.074120e-01f,  2.177420e-01f,  1.280720e-01f,  3.840200e-02f,
                9.386790e-01f,  8.486660e-01f,  7.586530e-01f,  6.686400e-01f,  5.786270e-01f,  4.886140e-01f,  3.986010e-01f,  3.085880e-01f,  2.185750e-01f,  1.285620e-01f,  3.854900e-02f,
                9.422560e-01f,  8.519000e-01f,  7.615440e-01f,  6.711880e-01f,  5.808320e-01f,  4.904760e-01f,  4.001200e-01f,  3.097640e-01f,  2.194080e-01f,  1.290520e-01f,  3.869600e-02f,
                9.458330e-01f,  8.551340e-01f,  7.644350e-01f,  6.737360e-01f,  5.830370e-01f,  4.923380e-01f,  4.016390e-01f,  3.109400e-01f,  2.202410e-01f,  1.295420e-01f,  3.884300e-02f,
                9.494100e-01f,  8.583680e-01f,  7.673260e-01f,  6.762840e-01f,  5.852420e-01f,  4.942000e-01f,  4.031580e-01f,  3.121160e-01f,  2.210740e-01f,  1.300320e-01f,  3.899000e-02f,
                9.529870e-01f,  8.616020e-01f,  7.702170e-01f,  6.788320e-01f,  5.874470e-01f,  4.960620e-01f,  4.046770e-01f,  3.132920e-01f,  2.219070e-01f,  1.305220e-01f,  3.913700e-02f,
                9.565640e-01f,  8.648360e-01f,  7.731080e-01f,  6.813800e-01f,  5.896520e-01f,  4.979240e-01f,  4.061960e-01f,  3.144680e-01f,  2.227400e-01f,  1.310120e-01f,  3.928400e-02f,
                9.601410e-01f,  8.680700e-01f,  7.759990e-01f,  6.839280e-01f,  5.918570e-01f,  4.997860e-01f,  4.077150e-01f,  3.156440e-01f,  2.235730e-01f,  1.315020e-01f,  3.943100e-02f,
                9.637180e-01f,  8.713040e-01f,  7.788900e-01f,  6.864760e-01f,  5.940620e-01f,  5.016480e-01f,  4.092340e-01f,  3.168200e-01f,  2.244060e-01f,  1.319920e-01f,  3.957800e-02f,
                9.672950e-01f,  8.745380e-01f,  7.817810e-01f,  6.890240e-01f,  5.962670e-01f,  5.035100e-01f,  4.107530e-01f,  3.179960e-01f,  2.252390e-01f,  1.324820e-01f,  3.972500e-02f,
                9.708720e-01f,  8.777720e-01f,  7.846720e-01f,  6.915720e-01f,  5.984720e-01f,  5.053720e-01f,  4.122720e-01f,  3.191720e-01f,  2.260720e-01f,  1.329720e-01f,  3.987200e-02f,
                9.744490e-01f,  8.810060e-01f,  7.875630e-01f,  6.941200e-01f,  6.006770e-01f,  5.072340e-01f,  4.137910e-01f,  3.203480e-01f,  2.269050e-01f,  1.334620e-01f,  4.001900e-02f,
                9.780260e-01f,  8.842400e-01f,  7.904540e-01f,  6.966680e-01f,  6.028820e-01f,  5.090960e-01f,  4.153100e-01f,  3.215240e-01f,  2.277380e-01f,  1.339520e-01f,  4.016600e-02f,
                9.816030e-01f,  8.874740e-01f,  7.933450e-01f,  6.992160e-01f,  6.050870e-01f,  5.109580e-01f,  4.168290e-01f,  3.227000e-01f,  2.285710e-01f,  1.344420e-01f,  4.031300e-02f,
                9.851800e-01f,  8.907080e-01f,  7.962360e-01f,  7.017640e-01f,  6.072920e-01f,  5.128200e-01f,  4.183480e-01f,  3.238760e-01f,  2.294040e-01f,  1.349320e-01f,  4.046000e-02f,
                9.887570e-01f,  8.939420e-01f,  7.991270e-01f,  7.043120e-01f,  6.094970e-01f,  5.146820e-01f,  4.198670e-01f,  3.250520e-01f,  2.302370e-01f,  1.354220e-01f,  4.060700e-02f,
                9.923340e-01f,  8.971760e-01f,  8.020180e-01f,  7.068600e-01f,  6.117020e-01f,  5.165440e-01f,  4.213860e-01f,  3.262280e-01f,  2.310700e-01f,  1.359120e-01f,  4.075400e-02f,
                9.959110e-01f,  9.004100e-01f,  8.049090e-01f,  7.094080e-01f,  6.139070e-01f,  5.184060e-01f,  4.229050e-01f,  3.274040e-01f,  2.319030e-01f,  1.364020e-01f,  4.090100e-02f,
                9.994880e-01f,  9.036440e-01f,  8.078000e-01f,  7.119560e-01f,  6.161120e-01f,  5.202680e-01f,  4.244240e-01f,  3.285800e-01f,  2.327360e-01f,  1.368920e-01f,  4.104800e-02f,
                1.003065e+00f,  9.068780e-01f,  8.106910e-01f,  7.145040e-01f,  6.183170e-01f,  5.221300e-01f,  4.259430e-01f,  3.297560e-01f,  2.335690e-01f,  1.373820e-01f,  4.119500e-02f,
                1.006642e+00f,  9.101120e-01f,  8.135820e-01f,  7.170520e-01f,  6.205220e-01f,  5.239920e-01f,  4.274620e-01f,  3.309320e-01f,  2.344020e-01f,  1.378720e-01f,  4.134200e-02f,
                1.010219e+00f,  9.133460e-01f,  8.164730e-01f,  7.196000e-01f,  6.227270e-01f,  5.258540e-01f,  4.289810e-01f,  3.321080e-01f,  2.352350e-01f,  1.383620e-01f,  4.148900e-02f,
                1.013796e+00f,  9.165800e-01f,  8.193640e-01f,  7.221480e-01f,  6.249320e-01f,  5.277160e-01f,  4.305000e-01f,  3.332840e-01f,  2.360680e-01f,  1.388520e-01f,  4.163600e-02f,
                1.017373e+00f,  9.198140e-01f,  8.222550e-01f,  7.246960e-01f,  6.271370e-01f,  5.295780e-01f,  4.320190e-01f,  3.344600e-01f,  2.369010e-01f,  1.393420e-01f,  4.178300e-02f,
                1.020950e+00f,  9.230480e-01f,  8.251460e-01f,  7.272440e-01f,  6.293420e-01f,  5.314400e-01f,  4.335380e-01f,  3.356360e-01f,  2.377340e-01f,  1.398320e-01f,  4.193000e-02f,
                1.024527e+00f,  9.262820e-01f,  8.280370e-01f,  7.297920e-01f,  6.315470e-01f,  5.333020e-01f,  4.350570e-01f,  3.368120e-01f,  2.385670e-01f,  1.403220e-01f,  4.207700e-02f,
                1.028104e+00f,  9.295160e-01f,  8.309280e-01f,  7.323400e-01f,  6.337520e-01f,  5.351640e-01f,  4.365760e-01f,  3.379880e-01f,  2.394000e-01f,  1.408120e-01f,  4.222400e-02f,
                1.031681e+00f,  9.327500e-01f,  8.338190e-01f,  7.348880e-01f,  6.359570e-01f,  5.370260e-01f,  4.380950e-01f,  3.391640e-01f,  2.402330e-01f,  1.413020e-01f,  4.237100e-02f,
                1.035258e+00f,  9.359840e-01f,  8.367100e-01f,  7.374360e-01f,  6.381620e-01f,  5.388880e-01f,  4.396140e-01f,  3.403400e-01f,  2.410660e-01f,  1.417920e-01f,  4.251800e-02f,
                1.038835e+00f,  9.392180e-01f,  8.396010e-01f,  7.399840e-01f,  6.403670e-01f,  5.407500e-01f,  4.411330e-01f,  3.415160e-01f,  2.418990e-01f,  1.422820e-01f,  4.266500e-02f,
                1.042412e+00f,  9.424520e-01f,  8.424920e-01f,  7.425320e-01f,  6.425720e-01f,  5.426120e-01f,  4.426520e-01f,  3.426920e-01f,  2.427320e-01f,  1.427720e-01f,  4.281200e-02f,
                1.045989e+00f,  9.456860e-01f,  8.453830e-01f,  7.450800e-01f,  6.447770e-01f,  5.444740e-01f,  4.441710e-01f,  3.438680e-01f,  2.435650e-01f,  1.432620e-01f,  4.295900e-02f,
                1.049566e+00f,  9.489200e-01f,  8.482740e-01f,  7.476280e-01f,  6.469820e-01f,  5.463360e-01f,  4.456900e-01f,  3.450440e-01f,  2.443980e-01f,  1.437520e-01f,  4.310600e-02f,
                1.053143e+00f,  9.521540e-01f,  8.511650e-01f,  7.501760e-01f,  6.491870e-01f,  5.481980e-01f,  4.472090e-01f,  3.462200e-01f,  2.452310e-01f,  1.442420e-01f,  4.325300e-02f,
                1.056720e+00f,  9.553880e-01f,  8.540560e-01f,  7.527240e-01f,  6.513920e-01f,  5.500600e-01f,  4.487280e-01f,  3.473960e-01f,  2.460640e-01f,  1.447320e-01f,  4.340000e-02f,
                1.060297e+00f,  9.586220e-01f,  8.569470e-01f,  7.552720e-01f,  6.535970e-01f,  5.519220e-01f,  4.502470e-01f,  3.485720e-01f,  2.468970e-01f,  1.452220e-01f,  4.354700e-02f,
                1.063874e+00f,  9.618560e-01f,  8.598380e-01f,  7.578200e-01f,  6.558020e-01f,  5.537840e-01f,  4.517660e-01f,  3.497480e-01f,  2.477300e-01f,  1.457120e-01f,  4.369400e-02f,
                1.067451e+00f,  9.650900e-01f,  8.627290e-01f,  7.603680e-01f,  6.580070e-01f,  5.556460e-01f,  4.532850e-01f,  3.509240e-01f,  2.485630e-01f,  1.462020e-01f,  4.384100e-02f,
                1.071028e+00f,  9.683240e-01f,  8.656200e-01f,  7.629160e-01f,  6.602120e-01f,  5.575080e-01f,  4.548040e-01f,  3.521000e-01f,  2.493960e-01f,  1.466920e-01f,  4.398800e-02f,
                1.074605e+00f,  9.715580e-01f,  8.685110e-01f,  7.654640e-01f,  6.624170e-01f,  5.593700e-01f,  4.563230e-01f,  3.532760e-01f,  2.502290e-01f,  1.471820e-01f,  4.413500e-02f,
                1.078182e+00f,  9.747920e-01f,  8.714020e-01f,  7.680120e-01f,  6.646220e-01f,  5.612320e-01f,  4.578420e-01f,  3.544520e-01f,  2.510620e-01f,  1.476720e-01f,  4.428200e-02f,
                1.081759e+00f,  9.780260e-01f,  8.742930e-01f,  7.705600e-01f,  6.668270e-01f,  5.630940e-01f,  4.593610e-01f,  3.556280e-01f,  2.518950e-01f,  1.481620e-01f,  4.442900e-02f,
                1.085336e+00f,  9.812600e-01f,  8.771840e-01f,  7.731080e-01f,  6.690320e-01f,  5.649560e-01f,  4.608800e-01f,  3.568040e-01f,  2.527280e-01f,  1.486520e-01f,  4.457600e-02f,
                1.088913e+00f,  9.844940e-01f,  8.800750e-01f,  7.756560e-01f,  6.712370e-01f,  5.668180e-01f,  4.623990e-01f,  3.579800e-01f,  2.535610e-01f,  1.491420e-01f,  4.472300e-02f,
                1.092490e+00f,  9.877280e-01f,  8.829660e-01f,  7.782040e-01f,  6.734420e-01f,  5.686800e-01f,  4.639180e-01f,  3.591560e-01f,  2.543940e-01f,  1.496320e-01f,  4.487000e-02f,
                1.096067e+00f,  9.909620e-01f,  8.858570e-01f,  7.807520e-01f,  6.756470e-01f,  5.705420e-01f,  4.654370e-01f,  3.603320e-01f,  2.552270e-01f,  1.501220e-01f,  4.501700e-02f,
                1.099644e+00f,  9.941960e-01f,  8.887480e-01f,  7.833000e-01f,  6.778520e-01f,  5.724040e-01f,  4.669560e-01f,  3.615080e-01f,  2.560600e-01f,  1.506120e-01f,  4.516400e-02f,
                1.103221e+00f,  9.974300e-01f,  8.916390e-01f,  7.858480e-01f,  6.800570e-01f,  5.742660e-01f,  4.684750e-01f,  3.626840e-01f,  2.568930e-01f,  1.511020e-01f,  4.531100e-02f,
                1.106798e+00f,  1.000664e+00f,  8.945300e-01f,  7.883960e-01f,  6.822620e-01f,  5.761280e-01f,  4.699940e-01f,  3.638600e-01f,  2.577260e-01f,  1.515920e-01f,  4.545800e-02f,
                1.110375e+00f,  1.003898e+00f,  8.974210e-01f,  7.909440e-01f,  6.844670e-01f,  5.779900e-01f,  4.715130e-01f,  3.650360e-01f,  2.585590e-01f,  1.520820e-01f,  4.560500e-02f,
                1.113952e+00f,  1.007132e+00f,  9.003120e-01f,  7.934920e-01f,  6.866720e-01f,  5.798520e-01f,  4.730320e-01f,  3.662120e-01f,  2.593920e-01f,  1.525720e-01f,  4.575200e-02f,
                1.117529e+00f,  1.010366e+00f,  9.032030e-01f,  7.960400e-01f,  6.888770e-01f,  5.817140e-01f,  4.745510e-01f,  3.673880e-01f,  2.602250e-01f,  1.530620e-01f,  4.589900e-02f,
                1.121106e+00f,  1.013600e+00f,  9.060940e-01f,  7.985880e-01f,  6.910820e-01f,  5.835760e-01f,  4.760700e-01f,  3.685640e-01f,  2.610580e-01f,  1.535520e-01f,  4.604600e-02f,
                1.124683e+00f,  1.016834e+00f,  9.089850e-01f,  8.011360e-01f,  6.932870e-01f,  5.854380e-01f,  4.775890e-01f,  3.697400e-01f,  2.618910e-01f,  1.540420e-01f,  4.619300e-02f,
                1.128260e+00f,  1.020068e+00f,  9.118760e-01f,  8.036840e-01f,  6.954920e-01f,  5.873000e-01f,  4.791080e-01f,  3.709160e-01f,  2.627240e-01f,  1.545320e-01f,  4.634000e-02f,
                1.131837e+00f,  1.023302e+00f,  9.147670e-01f,  8.062320e-01f,  6.976970e-01f,  5.891620e-01f,  4.806270e-01f,  3.720920e-01f,  2.635570e-01f,  1.550220e-01f,  4.648700e-02f,
                1.135414e+00f,  1.026536e+00f,  9.176580e-01f,  8.087800e-01f,  6.999020e-01f,  5.910240e-01f,  4.821460e-01f,  3.732680e-01f,  2.643900e-01f,  1.555120e-01f,  4.663400e-02f,
                1.138991e+00f,  1.029770e+00f,  9.205490e-01f,  8.113280e-01f,  7.021070e-01f,  5.928860e-01f,  4.836650e-01f,  3.744440e-01f,  2.652230e-01f,  1.560020e-01f,  4.678100e-02f,
                1.142568e+00f,  1.033004e+00f,  9.234400e-01f,  8.138760e-01f,  7.043120e-01f,  5.947480e-01f,  4.851840e-01f,  3.756200e-01f,  2.660560e-01f,  1.564920e-01f,  4.692800e-02f,
                1.146145e+00f,  1.036238e+00f,  9.263310e-01f,  8.164240e-01f,  7.065170e-01f,  5.966100e-01f,  4.867030e-01f,  3.767960e-01f,  2.668890e-01f,  1.569820e-01f,  4.707500e-02f,
                1.149722e+00f,  1.039472e+00f,  9.292220e-01f,  8.189720e-01f,  7.087220e-01f,  5.984720e-01f,  4.882220e-01f,  3.779720e-01f,  2.677220e-01f,  1.574720e-01f,  4.722200e-02f,
                1.153299e+00f,  1.042706e+00f,  9.321130e-01f,  8.215200e-01f,  7.109270e-01f,  6.003340e-01f,  4.897410e-01f,  3.791480e-01f,  2.685550e-01f,  1.579620e-01f,  4.736900e-02f,
                1.156876e+00f,  1.045940e+00f,  9.350040e-01f,  8.240680e-01f,  7.131320e-01f,  6.021960e-01f,  4.912600e-01f,  3.803240e-01f,  2.693880e-01f,  1.584520e-01f,  4.751600e-02f,
                1.160453e+00f,  1.049174e+00f,  9.378950e-01f,  8.266160e-01f,  7.153370e-01f,  6.040580e-01f,  4.927790e-01f,  3.815000e-01f,  2.702210e-01f,  1.589420e-01f,  4.766300e-02f,
                1.164030e+00f,  1.052408e+00f,  9.407860e-01f,  8.291640e-01f,  7.175420e-01f,  6.059200e-01f,  4.942980e-01f,  3.826760e-01f,  2.710540e-01f,  1.594320e-01f,  4.781000e-02f,
                1.167607e+00f,  1.055642e+00f,  9.436770e-01f,  8.317120e-01f,  7.197470e-01f,  6.077820e-01f,  4.958170e-01f,  3.838520e-01f,  2.718870e-01f,  1.599220e-01f,  4.795700e-02f,
                1.171184e+00f,  1.058876e+00f,  9.465680e-01f,  8.342600e-01f,  7.219520e-01f,  6.096440e-01f,  4.973360e-01f,  3.850280e-01f,  2.727200e-01f,  1.604120e-01f,  4.810400e-02f,
                1.174761e+00f,  1.062110e+00f,  9.494590e-01f,  8.368080e-01f,  7.241570e-01f,  6.115060e-01f,  4.988550e-01f,  3.862040e-01f,  2.735530e-01f,  1.609020e-01f,  4.825100e-02f,
                1.178338e+00f,  1.065344e+00f,  9.523500e-01f,  8.393560e-01f,  7.263620e-01f,  6.133680e-01f,  5.003740e-01f,  3.873800e-01f,  2.743860e-01f,  1.613920e-01f,  4.839800e-02f,
                1.181915e+00f,  1.068578e+00f,  9.552410e-01f,  8.419040e-01f,  7.285670e-01f,  6.152300e-01f,  5.018930e-01f,  3.885560e-01f,  2.752190e-01f,  1.618820e-01f,  4.854500e-02f,
                1.185492e+00f,  1.071812e+00f,  9.581320e-01f,  8.444520e-01f,  7.307720e-01f,  6.170920e-01f,  5.034120e-01f,  3.897320e-01f,  2.760520e-01f,  1.623720e-01f,  4.869200e-02f,
                1.189069e+00f,  1.075046e+00f,  9.610230e-01f,  8.470000e-01f,  7.329770e-01f,  6.189540e-01f,  5.049310e-01f,  3.909080e-01f,  2.768850e-01f,  1.628620e-01f,  4.883900e-02f,
                1.192646e+00f,  1.078280e+00f,  9.639140e-01f,  8.495480e-01f,  7.351820e-01f,  6.208160e-01f,  5.064500e-01f,  3.920840e-01f,  2.777180e-01f,  1.633520e-01f,  4.898600e-02f,
                1.196223e+00f,  1.081514e+00f,  9.668050e-01f,  8.520960e-01f,  7.373870e-01f,  6.226780e-01f,  5.079690e-01f,  3.932600e-01f,  2.785510e-01f,  1.638420e-01f,  4.913300e-02f,
                1.199800e+00f,  1.084748e+00f,  9.696960e-01f,  8.546440e-01f,  7.395920e-01f,  6.245400e-01f,  5.094880e-01f,  3.944360e-01f,  2.793840e-01f,  1.643320e-01f,  4.928000e-02f,
                1.203377e+00f,  1.087982e+00f,  9.725870e-01f,  8.571920e-01f,  7.417970e-01f,  6.264020e-01f,  5.110070e-01f,  3.956120e-01f,  2.802170e-01f,  1.648220e-01f,  4.942700e-02f,
                1.206954e+00f,  1.091216e+00f,  9.754780e-01f,  8.597400e-01f,  7.440020e-01f,  6.282640e-01f,  5.125260e-01f,  3.967880e-01f,  2.810500e-01f,  1.653120e-01f,  4.957400e-02f,
                1.210531e+00f,  1.094450e+00f,  9.783690e-01f,  8.622880e-01f,  7.462070e-01f,  6.301260e-01f,  5.140450e-01f,  3.979640e-01f,  2.818830e-01f,  1.658020e-01f,  4.972100e-02f,
                1.214108e+00f,  1.097684e+00f,  9.812600e-01f,  8.648360e-01f,  7.484120e-01f,  6.319880e-01f,  5.155640e-01f,  3.991400e-01f,  2.827160e-01f,  1.662920e-01f,  4.986800e-02f,
                1.217685e+00f,  1.100918e+00f,  9.841510e-01f,  8.673840e-01f,  7.506170e-01f,  6.338500e-01f,  5.170830e-01f,  4.003160e-01f,  2.835490e-01f,  1.667820e-01f,  5.001500e-02f,
                1.221262e+00f,  1.104152e+00f,  9.870420e-01f,  8.699320e-01f,  7.528220e-01f,  6.357120e-01f,  5.186020e-01f,  4.014920e-01f,  2.843820e-01f,  1.672720e-01f,  5.016200e-02f,
                1.224839e+00f,  1.107386e+00f,  9.899330e-01f,  8.724800e-01f,  7.550270e-01f,  6.375740e-01f,  5.201210e-01f,  4.026680e-01f,  2.852150e-01f,  1.677620e-01f,  5.030900e-02f,
                1.228416e+00f,  1.110620e+00f,  9.928240e-01f,  8.750280e-01f,  7.572320e-01f,  6.394360e-01f,  5.216400e-01f,  4.038440e-01f,  2.860480e-01f,  1.682520e-01f,  5.045600e-02f,
                1.231993e+00f,  1.113854e+00f,  9.957150e-01f,  8.775760e-01f,  7.594370e-01f,  6.412980e-01f,  5.231590e-01f,  4.050200e-01f,  2.868810e-01f,  1.687420e-01f,  5.060300e-02f,
                1.235570e+00f,  1.117088e+00f,  9.986060e-01f,  8.801240e-01f,  7.616420e-01f,  6.431600e-01f,  5.246780e-01f,  4.061960e-01f,  2.877140e-01f,  1.692320e-01f,  5.075000e-02f,
                1.239147e+00f,  1.120322e+00f,  1.001497e+00f,  8.826720e-01f,  7.638470e-01f,  6.450220e-01f,  5.261970e-01f,  4.073720e-01f,  2.885470e-01f,  1.697220e-01f,  5.089700e-02f,
                1.242724e+00f,  1.123556e+00f,  1.004388e+00f,  8.852200e-01f,  7.660520e-01f,  6.468840e-01f,  5.277160e-01f,  4.085480e-01f,  2.893800e-01f,  1.702120e-01f,  5.104400e-02f,
                1.246301e+00f,  1.126790e+00f,  1.007279e+00f,  8.877680e-01f,  7.682570e-01f,  6.487460e-01f,  5.292350e-01f,  4.097240e-01f,  2.902130e-01f,  1.707020e-01f,  5.119100e-02f,
                1.249878e+00f,  1.130024e+00f,  1.010170e+00f,  8.903160e-01f,  7.704620e-01f,  6.506080e-01f,  5.307540e-01f,  4.109000e-01f,  2.910460e-01f,  1.711920e-01f,  5.133800e-02f,
                1.253455e+00f,  1.133258e+00f,  1.013061e+00f,  8.928640e-01f,  7.726670e-01f,  6.524700e-01f,  5.322730e-01f,  4.120760e-01f,  2.918790e-01f,  1.716820e-01f,  5.148500e-02f,
                1.257032e+00f,  1.136492e+00f,  1.015952e+00f,  8.954120e-01f,  7.748720e-01f,  6.543320e-01f,  5.337920e-01f,  4.132520e-01f,  2.927120e-01f,  1.721720e-01f,  5.163200e-02f,
                1.260609e+00f,  1.139726e+00f,  1.018843e+00f,  8.979600e-01f,  7.770770e-01f,  6.561940e-01f,  5.353110e-01f,  4.144280e-01f,  2.935450e-01f,  1.726620e-01f,  5.177900e-02f,
                1.264186e+00f,  1.142960e+00f,  1.021734e+00f,  9.005080e-01f,  7.792820e-01f,  6.580560e-01f,  5.368300e-01f,  4.156040e-01f,  2.943780e-01f,  1.731520e-01f,  5.192600e-02f,
                1.267763e+00f,  1.146194e+00f,  1.024625e+00f,  9.030560e-01f,  7.814870e-01f,  6.599180e-01f,  5.383490e-01f,  4.167800e-01f,  2.952110e-01f,  1.736420e-01f,  5.207300e-02f,
                1.271340e+00f,  1.149428e+00f,  1.027516e+00f,  9.056040e-01f,  7.836920e-01f,  6.617800e-01f,  5.398680e-01f,  4.179560e-01f,  2.960440e-01f,  1.741320e-01f,  5.222000e-02f,
                1.274917e+00f,  1.152662e+00f,  1.030407e+00f,  9.081520e-01f,  7.858970e-01f,  6.636420e-01f,  5.413870e-01f,  4.191320e-01f,  2.968770e-01f,  1.746220e-01f,  5.236700e-02f,
                1.278494e+00f,  1.155896e+00f,  1.033298e+00f,  9.107000e-01f,  7.881020e-01f,  6.655040e-01f,  5.429060e-01f,  4.203080e-01f,  2.977100e-01f,  1.751120e-01f,  5.251400e-02f,
                1.282071e+00f,  1.159130e+00f,  1.036189e+00f,  9.132480e-01f,  7.903070e-01f,  6.673660e-01f,  5.444250e-01f,  4.214840e-01f,  2.985430e-01f,  1.756020e-01f,  5.266100e-02f,
                1.285648e+00f,  1.162364e+00f,  1.039080e+00f,  9.157960e-01f,  7.925120e-01f,  6.692280e-01f,  5.459440e-01f,  4.226600e-01f,  2.993760e-01f,  1.760920e-01f,  5.280800e-02f,
                1.289225e+00f,  1.165598e+00f,  1.041971e+00f,  9.183440e-01f,  7.947170e-01f,  6.710900e-01f,  5.474630e-01f,  4.238360e-01f,  3.002090e-01f,  1.765820e-01f,  5.295500e-02f,
                1.292802e+00f,  1.168832e+00f,  1.044862e+00f,  9.208920e-01f,  7.969220e-01f,  6.729520e-01f,  5.489820e-01f,  4.250120e-01f,  3.010420e-01f,  1.770720e-01f,  5.310200e-02f,
                1.296379e+00f,  1.172066e+00f,  1.047753e+00f,  9.234400e-01f,  7.991270e-01f,  6.748140e-01f,  5.505010e-01f,  4.261880e-01f,  3.018750e-01f,  1.775620e-01f,  5.324900e-02f,
                1.299956e+00f,  1.175300e+00f,  1.050644e+00f,  9.259880e-01f,  8.013320e-01f,  6.766760e-01f,  5.520200e-01f,  4.273640e-01f,  3.027080e-01f,  1.780520e-01f,  5.339600e-02f,
                1.303533e+00f,  1.178534e+00f,  1.053535e+00f,  9.285360e-01f,  8.035370e-01f,  6.785380e-01f,  5.535390e-01f,  4.285400e-01f,  3.035410e-01f,  1.785420e-01f,  5.354300e-02f,
                1.307110e+00f,  1.181768e+00f,  1.056426e+00f,  9.310840e-01f,  8.057420e-01f,  6.804000e-01f,  5.550580e-01f,  4.297160e-01f,  3.043740e-01f,  1.790320e-01f,  5.369000e-02f,
                1.310687e+00f,  1.185002e+00f,  1.059317e+00f,  9.336320e-01f,  8.079470e-01f,  6.822620e-01f,  5.565770e-01f,  4.308920e-01f,  3.052070e-01f,  1.795220e-01f,  5.383700e-02f,
                1.314264e+00f,  1.188236e+00f,  1.062208e+00f,  9.361800e-01f,  8.101520e-01f,  6.841240e-01f,  5.580960e-01f,  4.320680e-01f,  3.060400e-01f,  1.800120e-01f,  5.398400e-02f,
                1.317841e+00f,  1.191470e+00f,  1.065099e+00f,  9.387280e-01f,  8.123570e-01f,  6.859860e-01f,  5.596150e-01f,  4.332440e-01f,  3.068730e-01f,  1.805020e-01f,  5.413100e-02f,
                1.321418e+00f,  1.194704e+00f,  1.067990e+00f,  9.412760e-01f,  8.145620e-01f,  6.878480e-01f,  5.611340e-01f,  4.344200e-01f,  3.077060e-01f,  1.809920e-01f,  5.427800e-02f,
                1.324995e+00f,  1.197938e+00f,  1.070881e+00f,  9.438240e-01f,  8.167670e-01f,  6.897100e-01f,  5.626530e-01f,  4.355960e-01f,  3.085390e-01f,  1.814820e-01f,  5.442500e-02f,
                1.328572e+00f,  1.201172e+00f,  1.073772e+00f,  9.463720e-01f,  8.189720e-01f,  6.915720e-01f,  5.641720e-01f,  4.367720e-01f,  3.093720e-01f,  1.819720e-01f,  5.457200e-02f,
                1.332149e+00f,  1.204406e+00f,  1.076663e+00f,  9.489200e-01f,  8.211770e-01f,  6.934340e-01f,  5.656910e-01f,  4.379480e-01f,  3.102050e-01f,  1.824620e-01f,  5.471900e-02f,
                1.335726e+00f,  1.207640e+00f,  1.079554e+00f,  9.514680e-01f,  8.233820e-01f,  6.952960e-01f,  5.672100e-01f,  4.391240e-01f,  3.110380e-01f,  1.829520e-01f,  5.486600e-02f,
                1.339303e+00f,  1.210874e+00f,  1.082445e+00f,  9.540160e-01f,  8.255870e-01f,  6.971580e-01f,  5.687290e-01f,  4.403000e-01f,  3.118710e-01f,  1.834420e-01f,  5.501300e-02f,
                1.342880e+00f,  1.214108e+00f,  1.085336e+00f,  9.565640e-01f,  8.277920e-01f,  6.990200e-01f,  5.702480e-01f,  4.414760e-01f,  3.127040e-01f,  1.839320e-01f,  5.516000e-02f,
                1.346457e+00f,  1.217342e+00f,  1.088227e+00f,  9.591120e-01f,  8.299970e-01f,  7.008820e-01f,  5.717670e-01f,  4.426520e-01f,  3.135370e-01f,  1.844220e-01f,  5.530700e-02f,
                1.350034e+00f,  1.220576e+00f,  1.091118e+00f,  9.616600e-01f,  8.322020e-01f,  7.027440e-01f,  5.732860e-01f,  4.438280e-01f,  3.143700e-01f,  1.849120e-01f,  5.545400e-02f,
                1.353611e+00f,  1.223810e+00f,  1.094009e+00f,  9.642080e-01f,  8.344070e-01f,  7.046060e-01f,  5.748050e-01f,  4.450040e-01f,  3.152030e-01f,  1.854020e-01f,  5.560100e-02f,
                1.357188e+00f,  1.227044e+00f,  1.096900e+00f,  9.667560e-01f,  8.366120e-01f,  7.064680e-01f,  5.763240e-01f,  4.461800e-01f,  3.160360e-01f,  1.858920e-01f,  5.574800e-02f,
                1.360765e+00f,  1.230278e+00f,  1.099791e+00f,  9.693040e-01f,  8.388170e-01f,  7.083300e-01f,  5.778430e-01f,  4.473560e-01f,  3.168690e-01f,  1.863820e-01f,  5.589500e-02f,
                1.364342e+00f,  1.233512e+00f,  1.102682e+00f,  9.718520e-01f,  8.410220e-01f,  7.101920e-01f,  5.793620e-01f,  4.485320e-01f,  3.177020e-01f,  1.868720e-01f,  5.604200e-02f,
                1.367919e+00f,  1.236746e+00f,  1.105573e+00f,  9.744000e-01f,  8.432270e-01f,  7.120540e-01f,  5.808810e-01f,  4.497080e-01f,  3.185350e-01f,  1.873620e-01f,  5.618900e-02f,
                1.371496e+00f,  1.239980e+00f,  1.108464e+00f,  9.769480e-01f,  8.454320e-01f,  7.139160e-01f,  5.824000e-01f,  4.508840e-01f,  3.193680e-01f,  1.878520e-01f,  5.633600e-02f,
                1.375073e+00f,  1.243214e+00f,  1.111355e+00f,  9.794960e-01f,  8.476370e-01f,  7.157780e-01f,  5.839190e-01f,  4.520600e-01f,  3.202010e-01f,  1.883420e-01f,  5.648300e-02f,
                1.378650e+00f,  1.246448e+00f,  1.114246e+00f,  9.820440e-01f,  8.498420e-01f,  7.176400e-01f,  5.854380e-01f,  4.532360e-01f,  3.210340e-01f,  1.888320e-01f,  5.663000e-02f,
                1.382227e+00f,  1.249682e+00f,  1.117137e+00f,  9.845920e-01f,  8.520470e-01f,  7.195020e-01f,  5.869570e-01f,  4.544120e-01f,  3.218670e-01f,  1.893220e-01f,  5.677700e-02f,
                1.385804e+00f,  1.252916e+00f,  1.120028e+00f,  9.871400e-01f,  8.542520e-01f,  7.213640e-01f,  5.884760e-01f,  4.555880e-01f,  3.227000e-01f,  1.898120e-01f,  5.692400e-02f,
                1.389381e+00f,  1.256150e+00f,  1.122919e+00f,  9.896880e-01f,  8.564570e-01f,  7.232260e-01f,  5.899950e-01f,  4.567640e-01f,  3.235330e-01f,  1.903020e-01f,  5.707100e-02f,
                1.392958e+00f,  1.259384e+00f,  1.125810e+00f,  9.922360e-01f,  8.586620e-01f,  7.250880e-01f,  5.915140e-01f,  4.579400e-01f,  3.243660e-01f,  1.907920e-01f,  5.721800e-02f,
                1.396535e+00f,  1.262618e+00f,  1.128701e+00f,  9.947840e-01f,  8.608670e-01f,  7.269500e-01f,  5.930330e-01f,  4.591160e-01f,  3.251990e-01f,  1.912820e-01f,  5.736500e-02f,
                1.400112e+00f,  1.265852e+00f,  1.131592e+00f,  9.973320e-01f,  8.630720e-01f,  7.288120e-01f,  5.945520e-01f,  4.602920e-01f,  3.260320e-01f,  1.917720e-01f,  5.751200e-02f,
                1.403689e+00f,  1.269086e+00f,  1.134483e+00f,  9.998800e-01f,  8.652770e-01f,  7.306740e-01f,  5.960710e-01f,  4.614680e-01f,  3.268650e-01f,  1.922620e-01f,  5.765900e-02f,
                1.407266e+00f,  1.272320e+00f,  1.137374e+00f,  1.002428e+00f,  8.674820e-01f,  7.325360e-01f,  5.975900e-01f,  4.626440e-01f,  3.276980e-01f,  1.927520e-01f,  5.780600e-02f,
                1.410843e+00f,  1.275554e+00f,  1.140265e+00f,  1.004976e+00f,  8.696870e-01f,  7.343980e-01f,  5.991090e-01f,  4.638200e-01f,  3.285310e-01f,  1.932420e-01f,  5.795300e-02f,
                1.414420e+00f,  1.278788e+00f,  1.143156e+00f,  1.007524e+00f,  8.718920e-01f,  7.362600e-01f,  6.006280e-01f,  4.649960e-01f,  3.293640e-01f,  1.937320e-01f,  5.810000e-02f,
                1.417997e+00f,  1.282022e+00f,  1.146047e+00f,  1.010072e+00f,  8.740970e-01f,  7.381220e-01f,  6.021470e-01f,  4.661720e-01f,  3.301970e-01f,  1.942220e-01f,  5.824700e-02f,
                1.421574e+00f,  1.285256e+00f,  1.148938e+00f,  1.012620e+00f,  8.763020e-01f,  7.399840e-01f,  6.036660e-01f,  4.673480e-01f,  3.310300e-01f,  1.947120e-01f,  5.839400e-02f,
                1.425151e+00f,  1.288490e+00f,  1.151829e+00f,  1.015168e+00f,  8.785070e-01f,  7.418460e-01f,  6.051850e-01f,  4.685240e-01f,  3.318630e-01f,  1.952020e-01f,  5.854100e-02f,
                1.428728e+00f,  1.291724e+00f,  1.154720e+00f,  1.017716e+00f,  8.807120e-01f,  7.437080e-01f,  6.067040e-01f,  4.697000e-01f,  3.326960e-01f,  1.956920e-01f,  5.868800e-02f,
                1.432305e+00f,  1.294958e+00f,  1.157611e+00f,  1.020264e+00f,  8.829170e-01f,  7.455700e-01f,  6.082230e-01f,  4.708760e-01f,  3.335290e-01f,  1.961820e-01f,  5.883500e-02f,
                1.435882e+00f,  1.298192e+00f,  1.160502e+00f,  1.022812e+00f,  8.851220e-01f,  7.474320e-01f,  6.097420e-01f,  4.720520e-01f,  3.343620e-01f,  1.966720e-01f,  5.898200e-02f,
                1.439459e+00f,  1.301426e+00f,  1.163393e+00f,  1.025360e+00f,  8.873270e-01f,  7.492940e-01f,  6.112610e-01f,  4.732280e-01f,  3.351950e-01f,  1.971620e-01f,  5.912900e-02f,
                1.443036e+00f,  1.304660e+00f,  1.166284e+00f,  1.027908e+00f,  8.895320e-01f,  7.511560e-01f,  6.127800e-01f,  4.744040e-01f,  3.360280e-01f,  1.976520e-01f,  5.927600e-02f,
                1.446613e+00f,  1.307894e+00f,  1.169175e+00f,  1.030456e+00f,  8.917370e-01f,  7.530180e-01f,  6.142990e-01f,  4.755800e-01f,  3.368610e-01f,  1.981420e-01f,  5.942300e-02f,
                1.450190e+00f,  1.311128e+00f,  1.172066e+00f,  1.033004e+00f,  8.939420e-01f,  7.548800e-01f,  6.158180e-01f,  4.767560e-01f,  3.376940e-01f,  1.986320e-01f,  5.957000e-02f,
                1.453767e+00f,  1.314362e+00f,  1.174957e+00f,  1.035552e+00f,  8.961470e-01f,  7.567420e-01f,  6.173370e-01f,  4.779320e-01f,  3.385270e-01f,  1.991220e-01f,  5.971700e-02f,
                1.457344e+00f,  1.317596e+00f,  1.177848e+00f,  1.038100e+00f,  8.983520e-01f,  7.586040e-01f,  6.188560e-01f,  4.791080e-01f,  3.393600e-01f,  1.996120e-01f,  5.986400e-02f,
                1.460921e+00f,  1.320830e+00f,  1.180739e+00f,  1.040648e+00f,  9.005570e-01f,  7.604660e-01f,  6.203750e-01f,  4.802840e-01f,  3.401930e-01f,  2.001020e-01f,  6.001100e-02f,
                1.464498e+00f,  1.324064e+00f,  1.183630e+00f,  1.043196e+00f,  9.027620e-01f,  7.623280e-01f,  6.218940e-01f,  4.814600e-01f,  3.410260e-01f,  2.005920e-01f,  6.015800e-02f,
                1.468075e+00f,  1.327298e+00f,  1.186521e+00f,  1.045744e+00f,  9.049670e-01f,  7.641900e-01f,  6.234130e-01f,  4.826360e-01f,  3.418590e-01f,  2.010820e-01f,  6.030500e-02f,
                1.471652e+00f,  1.330532e+00f,  1.189412e+00f,  1.048292e+00f,  9.071720e-01f,  7.660520e-01f,  6.249320e-01f,  4.838120e-01f,  3.426920e-01f,  2.015720e-01f,  6.045200e-02f,
                1.475229e+00f,  1.333766e+00f,  1.192303e+00f,  1.050840e+00f,  9.093770e-01f,  7.679140e-01f,  6.264510e-01f,  4.849880e-01f,  3.435250e-01f,  2.020620e-01f,  6.059900e-02f,
                1.478806e+00f,  1.337000e+00f,  1.195194e+00f,  1.053388e+00f,  9.115820e-01f,  7.697760e-01f,  6.279700e-01f,  4.861640e-01f,  3.443580e-01f,  2.025520e-01f,  6.074600e-02f,
                1.482383e+00f,  1.340234e+00f,  1.198085e+00f,  1.055936e+00f,  9.137870e-01f,  7.716380e-01f,  6.294890e-01f,  4.873400e-01f,  3.451910e-01f,  2.030420e-01f,  6.089300e-02f,
                1.485960e+00f,  1.343468e+00f,  1.200976e+00f,  1.058484e+00f,  9.159920e-01f,  7.735000e-01f,  6.310080e-01f,  4.885160e-01f,  3.460240e-01f,  2.035320e-01f,  6.104000e-02f,
                1.489537e+00f,  1.346702e+00f,  1.203867e+00f,  1.061032e+00f,  9.181970e-01f,  7.753620e-01f,  6.325270e-01f,  4.896920e-01f,  3.468570e-01f,  2.040220e-01f,  6.118700e-02f,
                1.493114e+00f,  1.349936e+00f,  1.206758e+00f,  1.063580e+00f,  9.204020e-01f,  7.772240e-01f,  6.340460e-01f,  4.908680e-01f,  3.476900e-01f,  2.045120e-01f,  6.133400e-02f,
                1.496691e+00f,  1.353170e+00f,  1.209649e+00f,  1.066128e+00f,  9.226070e-01f,  7.790860e-01f,  6.355650e-01f,  4.920440e-01f,  3.485230e-01f,  2.050020e-01f,  6.148100e-02f,
                1.500268e+00f,  1.356404e+00f,  1.212540e+00f,  1.068676e+00f,  9.248120e-01f,  7.809480e-01f,  6.370840e-01f,  4.932200e-01f,  3.493560e-01f,  2.054920e-01f,  6.162800e-02f,
                1.503845e+00f,  1.359638e+00f,  1.215431e+00f,  1.071224e+00f,  9.270170e-01f,  7.828100e-01f,  6.386030e-01f,  4.943960e-01f,  3.501890e-01f,  2.059820e-01f,  6.177500e-02f,
                1.507422e+00f,  1.362872e+00f,  1.218322e+00f,  1.073772e+00f,  9.292220e-01f,  7.846720e-01f,  6.401220e-01f,  4.955720e-01f,  3.510220e-01f,  2.064720e-01f,  6.192200e-02f,
                1.510999e+00f,  1.366106e+00f,  1.221213e+00f,  1.076320e+00f,  9.314270e-01f,  7.865340e-01f,  6.416410e-01f,  4.967480e-01f,  3.518550e-01f,  2.069620e-01f,  6.206900e-02f,
                1.514576e+00f,  1.369340e+00f,  1.224104e+00f,  1.078868e+00f,  9.336320e-01f,  7.883960e-01f,  6.431600e-01f,  4.979240e-01f,  3.526880e-01f,  2.074520e-01f,  6.221600e-02f,
                1.518153e+00f,  1.372574e+00f,  1.226995e+00f,  1.081416e+00f,  9.358370e-01f,  7.902580e-01f,  6.446790e-01f,  4.991000e-01f,  3.535210e-01f,  2.079420e-01f,  6.236300e-02f,
                1.521730e+00f,  1.375808e+00f,  1.229886e+00f,  1.083964e+00f,  9.380420e-01f,  7.921200e-01f,  6.461980e-01f,  5.002760e-01f,  3.543540e-01f,  2.084320e-01f,  6.251000e-02f,
                1.525307e+00f,  1.379042e+00f,  1.232777e+00f,  1.086512e+00f,  9.402470e-01f,  7.939820e-01f,  6.477170e-01f,  5.014520e-01f,  3.551870e-01f,  2.089220e-01f,  6.265700e-02f,
                1.528884e+00f,  1.382276e+00f,  1.235668e+00f,  1.089060e+00f,  9.424520e-01f,  7.958440e-01f,  6.492360e-01f,  5.026280e-01f,  3.560200e-01f,  2.094120e-01f,  6.280400e-02f,
                1.532461e+00f,  1.385510e+00f,  1.238559e+00f,  1.091608e+00f,  9.446570e-01f,  7.977060e-01f,  6.507550e-01f,  5.038040e-01f,  3.568530e-01f,  2.099020e-01f,  6.295100e-02f,
                1.536038e+00f,  1.388744e+00f,  1.241450e+00f,  1.094156e+00f,  9.468620e-01f,  7.995680e-01f,  6.522740e-01f,  5.049800e-01f,  3.576860e-01f,  2.103920e-01f,  6.309800e-02f,
                1.539615e+00f,  1.391978e+00f,  1.244341e+00f,  1.096704e+00f,  9.490670e-01f,  8.014300e-01f,  6.537930e-01f,  5.061560e-01f,  3.585190e-01f,  2.108820e-01f,  6.324500e-02f,
                1.543192e+00f,  1.395212e+00f,  1.247232e+00f,  1.099252e+00f,  9.512720e-01f,  8.032920e-01f,  6.553120e-01f,  5.073320e-01f,  3.593520e-01f,  2.113720e-01f,  6.339200e-02f,
                1.546769e+00f,  1.398446e+00f,  1.250123e+00f,  1.101800e+00f,  9.534770e-01f,  8.051540e-01f,  6.568310e-01f,  5.085080e-01f,  3.601850e-01f,  2.118620e-01f,  6.353900e-02f,
                1.550346e+00f,  1.401680e+00f,  1.253014e+00f,  1.104348e+00f,  9.556820e-01f,  8.070160e-01f,  6.583500e-01f,  5.096840e-01f,  3.610180e-01f,  2.123520e-01f,  6.368600e-02f,
                1.553923e+00f,  1.404914e+00f,  1.255905e+00f,  1.106896e+00f,  9.578870e-01f,  8.088780e-01f,  6.598690e-01f,  5.108600e-01f,  3.618510e-01f,  2.128420e-01f,  6.383300e-02f,
                1.557500e+00f,  1.408148e+00f,  1.258796e+00f,  1.109444e+00f,  9.600920e-01f,  8.107400e-01f,  6.613880e-01f,  5.120360e-01f,  3.626840e-01f,  2.133320e-01f,  6.398000e-02f,
                1.561077e+00f,  1.411382e+00f,  1.261687e+00f,  1.111992e+00f,  9.622970e-01f,  8.126020e-01f,  6.629070e-01f,  5.132120e-01f,  3.635170e-01f,  2.138220e-01f,  6.412700e-02f,
                1.564654e+00f,  1.414616e+00f,  1.264578e+00f,  1.114540e+00f,  9.645020e-01f,  8.144640e-01f,  6.644260e-01f,  5.143880e-01f,  3.643500e-01f,  2.143120e-01f,  6.427400e-02f,
                1.568231e+00f,  1.417850e+00f,  1.267469e+00f,  1.117088e+00f,  9.667070e-01f,  8.163260e-01f,  6.659450e-01f,  5.155640e-01f,  3.651830e-01f,  2.148020e-01f,  6.442100e-02f,
                1.571808e+00f,  1.421084e+00f,  1.270360e+00f,  1.119636e+00f,  9.689120e-01f,  8.181880e-01f,  6.674640e-01f,  5.167400e-01f,  3.660160e-01f,  2.152920e-01f,  6.456800e-02f,
                1.575385e+00f,  1.424318e+00f,  1.273251e+00f,  1.122184e+00f,  9.711170e-01f,  8.200500e-01f,  6.689830e-01f,  5.179160e-01f,  3.668490e-01f,  2.157820e-01f,  6.471500e-02f,
                1.578962e+00f,  1.427552e+00f,  1.276142e+00f,  1.124732e+00f,  9.733220e-01f,  8.219120e-01f,  6.705020e-01f,  5.190920e-01f,  3.676820e-01f,  2.162720e-01f,  6.486200e-02f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{inwidth},{inheight},{batch}");
        }
    }
}
