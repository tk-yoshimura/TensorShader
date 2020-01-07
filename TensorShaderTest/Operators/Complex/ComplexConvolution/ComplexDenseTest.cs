using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ComplexConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Complex {
    [TestClass]
    public class ComplexDenseTest {
        [TestMethod]
        public void ExecuteTest() {
            Random random = new Random(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 2, 4, 10, 20, 32, 34 }) {
                    foreach (int outchannels in new int[] { 2, 4, 10, 20, 32, 34 }) {

                        float[] xval = (new float[inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                        float[] wval = (new float[inchannels * outchannels / 2]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                        System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                            .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

                        System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                            .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

                        ComplexMap0D x = new ComplexMap0D(inchannels / 2, batch, xcval);
                        ComplexFilter0D w = new ComplexFilter0D(inchannels / 2, outchannels / 2, wcval);

                        ComplexMap0D y = Reference(x, w);

                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch), xval);
                        OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels / 2), wval);

                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch));

                        ComplexDense ope = new ComplexDense(inchannels, outchannels, gradmode: false, batch);

                        ope.Execute(x_tensor, w_tensor, y_tensor);

                        float[] y_expect = y.ToArray();
                        float[] y_actual = y_tensor.State;

                        CollectionAssert.AreEqual(xval, x_tensor.State);
                        CollectionAssert.AreEqual(wval, w_tensor.State);

                        AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

                        Console.WriteLine($"pass: {inchannels},{outchannels},{batch}");
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void LargeMapTest() {
            Random random = new Random(1234);

            float max_err = 0;

            int batch = 3;
            int inchannels = 98, outchannels = 100;

            float[] xval = (new float[inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[inchannels * outchannels / 2]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

            ComplexMap0D x = new ComplexMap0D(inchannels / 2, batch, xcval);
            ComplexFilter0D w = new ComplexFilter0D(inchannels / 2, outchannels / 2, wcval);

            ComplexMap0D y = Reference(x, w);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch), xval);
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels / 2), wval);

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch));

            ComplexDense ope = new ComplexDense(inchannels, outchannels, gradmode: false, batch);

            ope.Execute(x_tensor, w_tensor, y_tensor);

            float[] y_expect = y.ToArray();
            float[] y_actual = y_tensor.State;

            CollectionAssert.AreEqual(xval, x_tensor.State);
            CollectionAssert.AreEqual(wval, w_tensor.State);

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inchannels = 32, outchannels = 32;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels / 2));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels));

            ComplexDense ope = new ComplexDense(inchannels, outchannels);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/complex_dense.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static ComplexMap0D Reference(ComplexMap0D x, ComplexFilter0D w) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;

            ComplexMap0D y = new ComplexMap0D(outchannels, batch);

            for (int th = 0; th < batch; th++) {
                for (int outch = 0; outch < outchannels; outch++) {
                    System.Numerics.Complex sum = y[outch, th];

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
            int inchannels = 6, outchannels = 8, batch = 3;

            float[] xval = (new float[batch * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

            ComplexMap0D x = new ComplexMap0D(inchannels / 2, batch, xcval);
            ComplexFilter0D w = new ComplexFilter0D(inchannels / 2, outchannels / 2, wcval);

            ComplexMap0D y = Reference(x, w);

            float[] y_expect = {
                -5.400000e-05f,  2.930000e-04f,  -3.600000e-05f,  2.030000e-04f,  -1.800000e-05f,  1.130000e-04f,
                0.000000e+00f,  2.300000e-05f,  -3.600000e-05f,  1.031000e-03f,  -1.800000e-05f,  7.250000e-04f,
                -2.710505e-20f,  4.190000e-04f,  1.800000e-05f,  1.130000e-04f,  -1.800000e-05f,  1.769000e-03f,
                0.000000e+00f,  1.247000e-03f,  1.800000e-05f,  7.250000e-04f,  3.600000e-05f,  2.030000e-04f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-9f, 1e-5f, $"mismatch value {inchannels},{outchannels},{batch}");
        }
    }
}
