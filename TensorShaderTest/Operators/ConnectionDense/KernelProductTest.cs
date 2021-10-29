using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.ConnectionDense;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.ConnectionDense {
    [TestClass]
    public class KernelProductTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                    foreach (int outchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                        float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                        float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                        Map0D x = new(inchannels, batch, xval);
                        Map0D y = new(outchannels, batch, yval);

                        Filter0D gw = Reference(x, y);

                        OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels, batch), xval);
                        OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels, batch), yval);

                        OverflowCheckedTensor gw_tensor = new(Shape.Kernel0D(inchannels, outchannels));

                        KernelProduct ope = new(inchannels, outchannels, batch);

                        ope.Execute(x_tensor, y_tensor, gw_tensor);

                        float[] gw_expect = gw.ToArray();
                        float[] gw_actual = gw_tensor.State.Value;

                        CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                        CollectionAssert.AreEqual(yval, y_tensor.State.Value);

                        AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

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
                        float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                        Map0D x = new(inchannels, batch, xval);
                        Map0D y = new(outchannels, batch, yval);

                        Filter0D gw = Reference(x, y);

                        OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels, batch), xval);
                        OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels, batch), yval);

                        OverflowCheckedTensor gw_tensor = new(Shape.Kernel0D(inchannels, outchannels));

                        KernelProduct ope = new(inchannels, outchannels, batch);

                        ope.Execute(x_tensor, y_tensor, gw_tensor);

                        float[] gw_expect = gw.ToArray();
                        float[] gw_actual = gw_tensor.State.Value;

                        CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                        CollectionAssert.AreEqual(yval, y_tensor.State.Value);

                        AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

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
            float[] yval = (new float[outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map0D x = new(inchannels, batch, xval);
            Map0D y = new(outchannels, batch, yval);

            Filter0D gw = Reference(x, y);

            OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels, batch), xval);
            OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels, batch), yval);

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel0D(inchannels, outchannels));

            KernelProduct ope = new(inchannels, outchannels, batch);

            ope.Execute(x_tensor, y_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(yval, y_tensor.State.Value);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inchannels = 1024, outchannels = 1024, batch = 4;

            OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels, batch));
            OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels, batch));

            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels));

            KernelProduct ope = new(inchannels, outchannels, batch);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/kernelproduct_dense_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, w_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inchannels = 1024, outchannels = 1024, batch = 4;

            OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels, batch));
            OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels, batch));

            OverflowCheckedTensor w_tensor = new(Shape.Kernel0D(inchannels, outchannels));

            KernelProduct ope = new(inchannels, outchannels, batch);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/kernelproduct_dense_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, w_tensor);

            Cuda.Profiler.Stop();
        }

        public static Filter0D Reference(Map0D x, Map0D y) {
            int inchannels = x.Channels, outchannels = y.Channels, batch = x.Batch;

            Filter0D w = new(inchannels, outchannels);

            for (int th = 0; th < batch; th++) {
                for (int inch, outch = 0; outch < outchannels; outch++) {
                    for (inch = 0; inch < inchannels; inch++) {
                        w[inch, outch] += x[inch, th] * y[outch, th];
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, batch = 2;

            float[] xval = (new float[batch * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[batch * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map0D x = new(inchannels, batch, xval);
            Map0D y = new(outchannels, batch, yval);

            Filter0D gw = Reference(x, y);

            float[] gw_expect = {
                7.0000e-05f, 1.0100e-04f, 1.3200e-04f, 1.6300e-04f, 1.9400e-04f,
                2.2500e-04f, 2.5600e-04f, 6.3000e-05f, 9.2000e-05f, 1.2100e-04f,
                1.5000e-04f, 1.7900e-04f, 2.0800e-04f, 2.3700e-04f, 5.6000e-05f,
                8.3000e-05f, 1.1000e-04f, 1.3700e-04f, 1.6400e-04f, 1.9100e-04f,
                2.1800e-04f, 4.9000e-05f, 7.4000e-05f, 9.9000e-05f, 1.2400e-04f,
                1.4900e-04f, 1.7400e-04f, 1.9900e-04f, 4.2000e-05f, 6.5000e-05f,
                8.8000e-05f, 1.1100e-04f, 1.3400e-04f, 1.5700e-04f, 1.8000e-04f,
                3.5000e-05f, 5.6000e-05f, 7.7000e-05f, 9.8000e-05f, 1.1900e-04f,
                1.4000e-04f, 1.6100e-04f, 2.8000e-05f, 4.7000e-05f, 6.6000e-05f,
                8.5000e-05f, 1.0400e-04f, 1.2300e-04f, 1.4200e-04f, 2.1000e-05f,
                3.8000e-05f, 5.5000e-05f, 7.2000e-05f, 8.9000e-05f, 1.0600e-04f,
                1.2300e-04f, 1.4000e-05f, 2.9000e-05f, 4.4000e-05f, 5.9000e-05f,
                7.4000e-05f, 8.9000e-05f, 1.0400e-04f, 7.0000e-06f, 2.0000e-05f,
                3.3000e-05f, 4.6000e-05f, 5.9000e-05f, 7.2000e-05f, 8.5000e-05f,
                0.0000e+00f, 1.1000e-05f, 2.2000e-05f, 3.3000e-05f, 4.4000e-05f,
                5.5000e-05f, 6.6000e-05f
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{batch}");
        }
    }
}
