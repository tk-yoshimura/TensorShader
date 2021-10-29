using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.TrivectorConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorConvolution1DTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                    foreach (int outchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                int outwidth = inwidth - kwidth + 1;

                                float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                float[] wval = (new float[kwidth * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                Trivector[] xcval = (new Trivector[xval.Length / 3])
                                    .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

                                Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                    .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                TrivectorMap1D x = new(inchannels / 3, inwidth, batch, xcval);
                                Quaternion.QuaternionFilter1D w = new(inchannels / 3, outchannels / 3, kwidth, wcval);

                                TrivectorMap1D y = Reference(x, w, kwidth);

                                OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch), xval);
                                OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth), wval);

                                OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth, batch));

                                TrivectorConvolution1D ope = new(inwidth, inchannels, outchannels, kwidth, gradmode: false, batch);

                                ope.Execute(x_tensor, w_tensor, y_tensor);

                                float[] y_expect = y.ToArray();
                                float[] y_actual = y_tensor.State.Value;

                                CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                AssertError.Tolerance(y_expect, y_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

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

            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                    foreach (int outchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                int outwidth = inwidth - kwidth + 1;

                                float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                                float[] wval = (new float[kwidth * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                                Trivector[] xcval = (new Trivector[xval.Length / 3])
                                    .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

                                Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                                    .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                                TrivectorMap1D x = new(inchannels / 3, inwidth, batch, xcval);
                                Quaternion.QuaternionFilter1D w = new(inchannels / 3, outchannels / 3, kwidth, wcval);

                                TrivectorMap1D y = Reference(x, w, kwidth);

                                OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch), xval);
                                OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth), wval);

                                OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth, batch));

                                TrivectorConvolution1D ope = new(inwidth, inchannels, outchannels, kwidth, gradmode: false, batch);

                                ope.Execute(x_tensor, w_tensor, y_tensor);

                                float[] y_expect = y.ToArray();
                                float[] y_actual = y_tensor.State.Value;

                                CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                                AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

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

            Random random = new(1234);

            float max_err = 0;

            int batch = 3;
            int inchannels = 147, outchannels = 150;
            int kwidth = 5;
            int inwidth = 125;
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Trivector[] xcval = (new Trivector[xval.Length / 3])
                .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap1D x = new(inchannels / 3, inwidth, batch, xcval);
            Quaternion.QuaternionFilter1D w = new(inchannels / 3, outchannels / 3, kwidth, wcval);

            TrivectorMap1D y = Reference(x, w, kwidth);

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth, batch), xval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth), wval);

            OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth, batch));

            TrivectorConvolution1D ope = new(inwidth, inchannels, outchannels, kwidth, gradmode: false, batch);

            ope.Execute(x_tensor, w_tensor, y_tensor);

            float[] y_expect = y.ToArray();
            float[] y_actual = y_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{inwidth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 32, inchannels = 33, outchannels = 33, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth));

            TrivectorConvolution1D ope = new(inwidth, inchannels, outchannels, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/trivector_convolution_1d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 32, inchannels = 33, outchannels = 33, ksize = 3;
            int outwidth = inwidth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map1D(outchannels, outwidth));

            TrivectorConvolution1D ope = new(inwidth, inchannels, outchannels, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/trivector_convolution_1d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static TrivectorMap1D Reference(TrivectorMap1D x, Quaternion.QuaternionFilter1D w, int kwidth) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width;
            int outw = inw - kwidth + 1;

            TrivectorMap1D y = new(outchannels, outw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        for (int outch = 0; outch < outchannels; outch++) {
                            Trivector sum = y[outch, ox, th];

                            for (int inch = 0; inch < inchannels; inch++) {
                                sum += x[inch, kx + ox, th] * w[inch, outch, kx];
                            }

                            y[outch, ox, th] = sum;
                        }
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 9, outchannels = 12, kwidth = 3, inwidth = 7, batch = 3;
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[batch * inwidth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Trivector[] xcval = (new Trivector[xval.Length / 3])
                .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap1D x = new(inchannels / 3, inwidth, batch, xcval);
            Quaternion.QuaternionFilter1D w = new(inchannels / 3, outchannels / 3, kwidth, wcval);

            TrivectorMap1D y = Reference(x, w, kwidth);

            float[] y_expect = {
                2.965368000e-03f,  2.182950000e-03f,  2.538060000e-03f,  2.200728000e-03f,  1.579446000e-03f,  1.861116000e-03f,
                1.581240000e-03f,  1.100358000e-03f,  1.318956000e-03f,  1.106904000e-03f,  7.456860000e-04f,  9.115800000e-04f,
                6.119886000e-03f,  5.249988000e-03f,  5.606070000e-03f,  4.698174000e-03f,  4.001076000e-03f,  4.283718000e-03f,
                3.514926000e-03f,  2.969892000e-03f,  3.189462000e-03f,  2.570142000e-03f,  2.156436000e-03f,  2.323302000e-03f,
                9.274404000e-03f,  8.317026000e-03f,  8.674080000e-03f,  7.195620000e-03f,  6.422706000e-03f,  6.706320000e-03f,
                5.448612000e-03f,  4.839426000e-03f,  5.059968000e-03f,  4.033380000e-03f,  3.567186000e-03f,  3.735024000e-03f,
                1.242892200e-02f,  1.138406400e-02f,  1.174209000e-02f,  9.693066000e-03f,  8.844336000e-03f,  9.128922000e-03f,
                7.382298000e-03f,  6.708960000e-03f,  6.930474000e-03f,  5.496618000e-03f,  4.977936000e-03f,  5.146746000e-03f,
                1.558344000e-02f,  1.445110200e-02f,  1.481010000e-02f,  1.219051200e-02f,  1.126596600e-02f,  1.155152400e-02f,
                9.315984000e-03f,  8.578494000e-03f,  8.800980000e-03f,  6.959856000e-03f,  6.388686000e-03f,  6.558468000e-03f,
                2.504699400e-02f,  2.365221600e-02f,  2.401413000e-02f,  1.968285000e-02f,  1.853085600e-02f,  1.881933000e-02f,
                1.511704200e-02f,  1.418709600e-02f,  1.441249800e-02f,  1.134957000e-02f,  1.062093600e-02f,  1.079363400e-02f,
                2.820151200e-02f,  2.671925400e-02f,  2.708214000e-02f,  2.218029600e-02f,  2.095248600e-02f,  2.124193200e-02f,
                1.705072800e-02f,  1.605663000e-02f,  1.628300400e-02f,  1.281280800e-02f,  1.203168600e-02f,  1.220535600e-02f,
                3.135603000e-02f,  2.978629200e-02f,  3.015015000e-02f,  2.467774200e-02f,  2.337411600e-02f,  2.366453400e-02f,
                1.898441400e-02f,  1.792616400e-02f,  1.815351000e-02f,  1.427604600e-02f,  1.344243600e-02f,  1.361707800e-02f,
                3.451054800e-02f,  3.285333000e-02f,  3.321816000e-02f,  2.717518800e-02f,  2.579574600e-02f,  2.608713600e-02f,
                2.091810000e-02f,  1.979569800e-02f,  2.002401600e-02f,  1.573928400e-02f,  1.485318600e-02f,  1.502880000e-02f,
                3.766506600e-02f,  3.592036800e-02f,  3.628617000e-02f,  2.967263400e-02f,  2.821737600e-02f,  2.850973800e-02f,
                2.285178600e-02f,  2.166523200e-02f,  2.189452200e-02f,  1.720252200e-02f,  1.626393600e-02f,  1.644052200e-02f,
                4.712862000e-02f,  4.512148200e-02f,  4.549020000e-02f,  3.716497200e-02f,  3.548226600e-02f,  3.577754400e-02f,
                2.865284400e-02f,  2.727383400e-02f,  2.750604000e-02f,  2.159223600e-02f,  2.049618600e-02f,  2.067568800e-02f,
                5.028313800e-02f,  4.818852000e-02f,  4.855821000e-02f,  3.966241800e-02f,  3.790389600e-02f,  3.820014600e-02f,
                3.058653000e-02f,  2.914336800e-02f,  2.937654600e-02f,  2.305547400e-02f,  2.190693600e-02f,  2.208741000e-02f,
                5.343765600e-02f,  5.125555800e-02f,  5.162622000e-02f,  4.215986400e-02f,  4.032552600e-02f,  4.062274800e-02f,
                3.252021600e-02f,  3.101290200e-02f,  3.124705200e-02f,  2.451871200e-02f,  2.331768600e-02f,  2.349913200e-02f,
                5.659217400e-02f,  5.432259600e-02f,  5.469423000e-02f,  4.465731000e-02f,  4.274715600e-02f,  4.304535000e-02f,
                3.445390200e-02f,  3.288243600e-02f,  3.311755800e-02f,  2.598195000e-02f,  2.472843600e-02f,  2.491085400e-02f,
                5.974669200e-02f,  5.738963400e-02f,  5.776224000e-02f,  4.715475600e-02f,  4.516878600e-02f,  4.546795200e-02f,
                3.638758800e-02f,  3.475197000e-02f,  3.498806400e-02f,  2.744518800e-02f,  2.613918600e-02f,  2.632257600e-02f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");
        }
    }
}
