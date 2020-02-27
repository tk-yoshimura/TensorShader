using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.TrivectorConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorTransposeDenseTest {
        [TestMethod]
        public void ExecuteTest() {
            Random random = new Random(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {
                    foreach (int outchannels in new int[] { 3, 6, 9, 15, 21, 33 }) {

                        float[] yval = (new float[outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                        float[] wval = (new float[inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                        Trivector[] ycval = (new Trivector[yval.Length / 3])
                            .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

                        Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                            .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                        TrivectorMap0D y = new TrivectorMap0D(outchannels / 3, batch, ycval);
                        Quaternion.QuaternionFilter0D w = new Quaternion.QuaternionFilter0D(inchannels / 3, outchannels / 3, wcval);

                        TrivectorMap0D x = Reference(y, w);

                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch), yval);
                        OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3), wval);

                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch));

                        TrivectorTransposeDense ope = new TrivectorTransposeDense(outchannels, inchannels, gradmode: false, batch);

                        ope.Execute(y_tensor, w_tensor, x_tensor);

                        float[] x_expect = x.ToArray();
                        float[] x_actual = x_tensor.State;

                        CollectionAssert.AreEqual(yval, y_tensor.State);
                        CollectionAssert.AreEqual(wval, w_tensor.State);

                        AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

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
            int inchannels = 147, outchannels = 150;

            float[] yval = (new float[outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[inchannels * outchannels / 9 * 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Trivector[] ycval = (new Trivector[yval.Length / 3])
                .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap0D y = new TrivectorMap0D(outchannels / 3, batch, ycval);
            Quaternion.QuaternionFilter0D w = new Quaternion.QuaternionFilter0D(inchannels / 3, outchannels / 3, wcval);

            TrivectorMap0D x = Reference(y, w);

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch), yval);
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3), wval);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch));

            TrivectorTransposeDense ope = new TrivectorTransposeDense(outchannels, inchannels, gradmode: false, batch);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            float[] x_expect = x.ToArray();
            float[] x_actual = x_tensor.State;

            CollectionAssert.AreEqual(yval, y_tensor.State);
            CollectionAssert.AreEqual(wval, w_tensor.State);

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inchannels = 33, outchannels = 33;

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3));

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels));

            TrivectorTransposeDense ope = new TrivectorTransposeDense(outchannels, inchannels);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/trivector_transpose_dense.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        public static TrivectorMap0D Reference(TrivectorMap0D y, Quaternion.QuaternionFilter0D w) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;

            TrivectorMap0D x = new TrivectorMap0D(inchannels, batch);

            for (int th = 0; th < batch; th++) {
                for (int outch = 0; outch < outchannels; outch++) {
                    Trivector v = y[outch, th];

                    for (int inch = 0; inch < inchannels; inch++) {
                        x[inch, th] += v * w[inch, outch];
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 9, outchannels = 12, batch = 3;

            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-2f).ToArray();
            float[] wval = (new float[outchannels * inchannels / 9 * 4]).Select((_, idx) => idx * 1e-2f).Reverse().ToArray();

            Trivector[] ycval = (new Trivector[yval.Length / 3])
                .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

            Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            TrivectorMap0D y = new TrivectorMap0D(outchannels / 3, batch, ycval);
            Quaternion.QuaternionFilter0D w = new Quaternion.QuaternionFilter0D(inchannels / 3, outchannels / 3, wcval);

            TrivectorMap0D x = Reference(y, w);

            float[] x_expect = {
                6.030000000e-02f,  2.514000000e-02f,  4.170000000e-02f,  4.426800000e-02f,  1.669200000e-02f,  2.979600000e-02f,
                3.156400000e-02f,  1.054800000e-02f,  2.070800000e-02f,  2.506680000e-01f,  1.993800000e-01f,  2.165160000e-01f,
                1.939320000e-01f,  1.525320000e-01f,  1.662120000e-01f,  1.466680000e-01f,  1.141320000e-01f,  1.248680000e-01f,
                4.410360000e-01f,  3.736200000e-01f,  3.913320000e-01f,  3.435960000e-01f,  2.883720000e-01f,  3.026280000e-01f,
                2.617720000e-01f,  2.177160000e-01f,  2.290280000e-01f
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{batch}");
        }
    }
}
