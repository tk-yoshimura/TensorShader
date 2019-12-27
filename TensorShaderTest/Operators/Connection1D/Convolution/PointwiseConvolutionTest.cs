using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class PointwiseConvolutionTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        foreach (int width in new int[] { 8, 9, 13, 17 }) {
                            float[] xval = (new float[width * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map1D x = new Map1D(inchannels, width, batch, xval);
                            Filter1D w = new Filter1D(inchannels, outchannels, 1, wval);

                            Map1D y = Reference(x, w);

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, width, batch), xval);
                            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels), wval);

                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, width, batch));

                            PointwiseConvolution ope = new PointwiseConvolution(width, inchannels, outchannels, batch);

                            ope.Execute(x_tensor, w_tensor, y_tensor);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_tensor.State;

                            CollectionAssert.AreEqual(xval, x_tensor.State);
                            CollectionAssert.AreEqual(wval, w_tensor.State);

                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{width},{batch}");

                            Console.WriteLine($"pass: {inchannels},{outchannels},{width},{batch}");
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
            int width = 128;

            float[] xval = (new float[width * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map1D x = new Map1D(inchannels, width, batch, xval);
            Filter1D w = new Filter1D(inchannels, outchannels, 1, wval);

            Map1D y = Reference(x, w);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, width, batch), xval);
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels), wval);

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, width, batch));

            PointwiseConvolution ope = new PointwiseConvolution(width, inchannels, outchannels, batch);

            ope.Execute(x_tensor, w_tensor, y_tensor);

            float[] y_expect = y.ToArray();
            float[] y_actual = y_tensor.State;

            CollectionAssert.AreEqual(xval, x_tensor.State);
            CollectionAssert.AreEqual(wval, w_tensor.State);

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{width},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{width},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inchannels = 31, outchannels = 63;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, inwidth));

            PointwiseConvolution ope = new PointwiseConvolution(inwidth, inchannels, outchannels);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/ptwise_convolution_1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);
            
            Cuda.Profiler.Stop();
        }

        public static Map1D Reference(Map1D x, Filter1D w) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width;

            Map1D y = new Map1D(outchannels, inw, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix = 0; ix < inw; ix++) {
                    for (int outch = 0; outch < outchannels; outch++) {
                        double sum = y[outch, ix, th];

                        for (int inch = 0; inch < inchannels; inch++) {
                            sum += x[inch, ix, th] * w[inch, outch, 0];
                        }

                        y[outch, ix, th] = sum;
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, inwidth = 13, batch = 2;

            float[] xval = (new float[batch * inwidth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map1D x = new Map1D(inchannels, inwidth, batch, xval);
            Filter1D w = new Filter1D(inchannels, outchannels, 1, wval);

            Map1D y = Reference(x, w);

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
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{inwidth},{batch}");
        }
    }
}
