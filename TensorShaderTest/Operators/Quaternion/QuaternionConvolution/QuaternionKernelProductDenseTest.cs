using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.QuaternionConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Quaternion {
    [TestClass]
    public class QuaternionKernelProductDenseTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 4, 8, 20, 32, 36 }) {
                    foreach (int outchannels in new int[] { 4, 8, 20, 32, 36 }) {

                        float[] xval = (new float[inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                        float[] yval = (new float[outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                        Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                            .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

                        Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                            .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

                        QuaternionMap0D x = new(inchannels / 4, batch, xcval);
                        QuaternionMap0D y = new(outchannels / 4, batch, ycval);

                        QuaternionFilter0D gw = Reference(x, y);

                        OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels, batch), xval);
                        OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels, batch), yval);

                        OverflowCheckedTensor gw_tensor = new(Shape.Kernel0D(inchannels, outchannels / 4));

                        QuaternionKernelProductDense ope = new(inchannels, outchannels, transpose: false, batch);

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

            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 4, 8, 20, 32, 36 }) {
                    foreach (int outchannels in new int[] { 4, 8, 20, 32, 36 }) {

                        float[] xval = (new float[inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                        float[] yval = (new float[outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                        Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                            .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

                        Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                            .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

                        QuaternionMap0D x = new(inchannels / 4, batch, xcval);
                        QuaternionMap0D y = new(outchannels / 4, batch, ycval);

                        QuaternionFilter0D gw = Reference(x, y);

                        OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels, batch), xval);
                        OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels, batch), yval);

                        OverflowCheckedTensor gw_tensor = new(Shape.Kernel0D(inchannels, outchannels / 4));

                        QuaternionKernelProductDense ope = new(inchannels, outchannels, transpose: false, batch);

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

            Random random = new(1234);

            float max_err = 0;

            int batch = 3;
            int inchannels = 196, outchannels = 200;

            float[] xval = (new float[inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] yval = (new float[outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            QuaternionMap0D x = new(inchannels / 4, batch, xcval);
            QuaternionMap0D y = new(outchannels / 4, batch, ycval);

            QuaternionFilter0D gw = Reference(x, y);

            OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels, batch), xval);
            OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels, batch), yval);

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel0D(inchannels, outchannels / 4));

            QuaternionKernelProductDense ope = new(inchannels, outchannels, transpose: false, batch);

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

            int inchannels = 32, outchannels = 32;

            OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels));
            OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel0D(inchannels, outchannels / 4));

            QuaternionKernelProductDense ope = new(inchannels, outchannels);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/quaternion_kernelproduct_dense_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inchannels = 32, outchannels = 32;

            OverflowCheckedTensor x_tensor = new(Shape.Map0D(inchannels));
            OverflowCheckedTensor y_tensor = new(Shape.Map0D(outchannels));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel0D(inchannels, outchannels / 4));

            QuaternionKernelProductDense ope = new(inchannels, outchannels);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/quaternion_kernelproduct_dense_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static QuaternionFilter0D Reference(QuaternionMap0D x, QuaternionMap0D gy) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;

            QuaternionFilter0D w = new(inchannels, outchannels);

            for (int th = 0; th < batch; th++) {
                for (int inch, outch = 0; outch < outchannels; outch++) {
                    for (inch = 0; inch < inchannels; inch++) {
                        w[inch, outch] += Quaternion.MulGrad(gy[outch, th], x[inch, th]);
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 8, outchannels = 12, batch = 3;

            float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            QuaternionMap0D x = new(inchannels / 4, batch, xcval);
            QuaternionMap0D y = new(outchannels / 4, batch, ycval);

            QuaternionFilter0D gw = Reference(x, y);

            float[] gw_expect = {
                1.668000e-03f,  -1.084202e-19f,  -3.720000e-04f,  -1.860000e-04f,  2.700000e-03f,  1.084202e-19f,
                -4.200000e-04f,  -2.100000e-04f,  1.212000e-03f,  -0.000000e+00f,  -3.240000e-04f,  -1.620000e-04f,
                2.052000e-03f,  1.084202e-19f,  -3.720000e-04f,  -1.860000e-04f,  7.560000e-04f,  -0.000000e+00f,
                -2.760000e-04f,  -1.380000e-04f,  1.404000e-03f,  5.421011e-20f,  -3.240000e-04f,  -1.620000e-04f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{batch}");
        }
    }
}
