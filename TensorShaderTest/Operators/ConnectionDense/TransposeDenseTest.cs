using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.ConnectionDense;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.ConnectionDense {
    [TestClass]
    public class TransposeDenseTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                    foreach (int outchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                        float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                        float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                        Map0D y = new(outchannels, batch, yval);
                        Filter0D w = new(inchannels, outchannels, wval);

                        Map0D x = Reference(y, w);

                        OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels, batch), yval);
                        OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

                        OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels, batch));

                        TransposeDense ope = new(outchannels, inchannels, batch);

                        ope.Execute(y_tensor, w_tensor, x_tensor);

                        float[] x_expect = x.ToArray();
                        float[] x_actual = x_tensor.State.Value;

                        CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                        CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                        AssertError.Tolerance(x_expect, x_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

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
                        float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                        float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                        Map0D y = new(outchannels, batch, yval);
                        Filter0D w = new(inchannels, outchannels, wval);

                        Map0D x = Reference(y, w);

                        OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels, batch), yval);
                        OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

                        OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels, batch));

                        TransposeDense ope = new(outchannels, inchannels, batch);

                        ope.Execute(y_tensor, w_tensor, x_tensor);

                        float[] x_expect = x.ToArray();
                        float[] x_actual = x_tensor.State.Value;

                        CollectionAssert.AreEqual(yval, y_tensor.State.Value);
                        CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                        AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

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

            float[] yval = (new float[outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map0D y = new(outchannels, batch, yval);
            Filter0D w = new(inchannels, outchannels, wval);

            Map0D x = Reference(y, w);

            OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels, batch), yval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels), wval);

            OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels, batch));

            TransposeDense ope = new(outchannels, inchannels, batch);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            float[] x_expect = x.ToArray();
            float[] x_actual = x_tensor.State.Value;

            CollectionAssert.AreEqual(yval, y_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inchannels = 1024, outchannels = 1024;

            OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels));

            TransposeDense ope = new(outchannels, inchannels);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/transpose_dense_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inchannels = 1024, outchannels = 1024;

            OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels));

            TransposeDense ope = new(outchannels, inchannels);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/transpose_dense_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map0D Reference(Map0D y, Filter0D w) {
            int outchannels = y.Channels, inchannels = w.InChannels, batch = y.Batch;

            Map0D x = new(inchannels, batch);

            for (int th = 0; th < batch; th++) {
                for (int inch = 0; inch < inchannels; inch++) {
                    double sum = 0;

                    for (int outch = 0; outch < outchannels; outch++) {
                        sum += y[outch, th] * w[inch, outch];
                    }

                    x[inch, th] = sum;
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, batch = 2;

            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map0D y = new(outchannels, batch, yval);
            Filter0D w = new(inchannels, outchannels, wval);

            Map0D x = Reference(y, w);

            float[] x_expect = {
                1.4850e-03f, 1.4300e-03f, 1.3750e-03f, 1.3200e-03f, 1.2650e-03f,
                1.2100e-03f, 1.1550e-03f, 6.4460e-03f, 6.2700e-03f, 6.0940e-03f,
                5.9180e-03f, 5.7420e-03f, 5.5660e-03f, 5.3900e-03f
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{batch}");
        }
    }
}
