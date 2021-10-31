using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.ConnectionDense;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.ConnectionDense {
    [TestClass]
    public class DenseTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                    foreach (int outchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                        float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                        float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                        Map0D x = new(inchannels, batch, xval);
                        Filter0D w = new(inchannels, outchannels, wval);

                        Map0D y = Reference(x, w);

                        OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels, batch), xval);
                        OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

                        OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels, batch));

                        Dense ope = new(inchannels, outchannels, batch);

                        ope.Execute(x_tensor, w_tensor, y_tensor);

                        float[] y_expect = y.ToArray();
                        float[] y_actual = y_tensor.State.Value;

                        CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                        CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                        AssertError.Tolerance(y_expect, y_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

                        Console.WriteLine($"pass: {inchannels},{outchannels},{batch}");
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
                        float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                        float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                        Map0D x = new(inchannels, batch, xval);
                        Filter0D w = new(inchannels, outchannels, wval);

                        Map0D y = Reference(x, w);

                        OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels, batch), xval);
                        OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

                        OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels, batch));

                        Dense ope = new(inchannels, outchannels, batch);

                        ope.Execute(x_tensor, w_tensor, y_tensor);

                        float[] y_expect = y.ToArray();
                        float[] y_actual = y_tensor.State.Value;

                        CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                        CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                        AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

                        Console.WriteLine($"pass: {inchannels},{outchannels},{batch}");
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
                        float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                        float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                        Map0D x = new(inchannels, batch, xval);
                        Filter0D w = new(inchannels, outchannels, wval);

                        Map0D y = Reference(x, w);

                        OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels, batch), xval);
                        OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

                        OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels, batch));

                        Dense ope = new(inchannels, outchannels, batch);

                        ope.Execute(x_tensor, w_tensor, y_tensor);

                        float[] y_expect = y.ToArray();
                        float[] y_actual = y_tensor.State.Value;

                        CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                        CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                        AssertError.Tolerance(y_expect, y_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

                        Console.WriteLine($"pass: {inchannels},{outchannels},{batch}");
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

            float[] xval = (new float[inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map0D x = new(inchannels, batch, xval);
            Filter0D w = new(inchannels, outchannels, wval);

            Map0D y = Reference(x, w);

            OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels, batch), xval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

            OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels, batch));

            Dense ope = new(inchannels, outchannels, batch);

            ope.Execute(x_tensor, w_tensor, y_tensor);

            float[] y_expect = y.ToArray();
            float[] y_actual = y_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inchannels = 1024, outchannels = 1024;

            OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels));

            Dense ope = new(inchannels, outchannels);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/dense_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inchannels = 1024, outchannels = 1024;

            OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels));

            Dense ope = new(inchannels, outchannels);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/dense_ffp.nvvp");
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

            int inchannels = 1024, outchannels = 1024;

            OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels));

            Dense ope = new(inchannels, outchannels);

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/dense_cudnn.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map0D Reference(Map0D x, Filter0D w) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;

            Map0D y = new(outchannels, batch);

            for (int th = 0; th < batch; th++) {
                for (int outch = 0; outch < outchannels; outch++) {
                    double sum = 0;

                    for (int inch = 0; inch < inchannels; inch++) {
                        sum += x[inch, th] * w[inch, outch];
                    }

                    y[outch, th] = sum;
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, batch = 2;

            float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map0D x = new(inchannels, batch, xval);
            Filter0D w = new(inchannels, outchannels, wval);

            Map0D y = Reference(x, w);

            float[] y_expect = {
                1.5050e-03f, 1.3580e-03f, 1.2110e-03f, 1.0640e-03f, 9.1700e-04f,
                7.7000e-04f, 6.2300e-04f, 4.7600e-04f, 3.2900e-04f, 1.8200e-04f,
                3.5000e-05f, 5.0820e-03f, 4.5920e-03f, 4.1020e-03f, 3.6120e-03f,
                3.1220e-03f, 2.6320e-03f, 2.1420e-03f, 1.6520e-03f, 1.1620e-03f,
                6.7200e-04f, 1.8200e-04f
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{batch}");
        }
    }
}
