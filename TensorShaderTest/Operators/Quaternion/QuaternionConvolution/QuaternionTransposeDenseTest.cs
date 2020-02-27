using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.QuaternionConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Quaternion {
    [TestClass]
    public class QuaternionTransposeDenseTest {
        [TestMethod]
        public void ExecuteTest() {
            Random random = new Random(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 4, 8, 20, 32, 36 }) {
                    foreach (int outchannels in new int[] { 4, 8, 20, 32, 36 }) {

                        float[] yval = (new float[outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                        float[] wval = (new float[inchannels * outchannels / 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                        Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                            .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

                        Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                            .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                        QuaternionMap0D y = new QuaternionMap0D(outchannels / 4, batch, ycval);
                        QuaternionFilter0D w = new QuaternionFilter0D(inchannels / 4, outchannels / 4, wcval);

                        QuaternionMap0D x = Reference(y, w);

                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch), yval);
                        OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels / 4), wval);

                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch));

                        QuaternionTransposeDense ope = new QuaternionTransposeDense(outchannels, inchannels, gradmode: false, batch);

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
            int inchannels = 196, outchannels = 200;

            float[] yval = (new float[outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[inchannels * outchannels / 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            QuaternionMap0D y = new QuaternionMap0D(outchannels / 4, batch, ycval);
            QuaternionFilter0D w = new QuaternionFilter0D(inchannels / 4, outchannels / 4, wcval);

            QuaternionMap0D x = Reference(y, w);

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch), yval);
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels / 4), wval);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch));

            QuaternionTransposeDense ope = new QuaternionTransposeDense(outchannels, inchannels, gradmode: false, batch);

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
            int inchannels = 32, outchannels = 32;

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels / 4));

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels));

            QuaternionTransposeDense ope = new QuaternionTransposeDense(outchannels, inchannels);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/quaternion_transpose_dense.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        public static QuaternionMap0D Reference(QuaternionMap0D y, QuaternionFilter0D w) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;

            QuaternionMap0D x = new QuaternionMap0D(inchannels, batch);

            for (int th = 0; th < batch; th++) {
                for (int outch = 0; outch < outchannels; outch++) {
                    Quaternion v = y[outch, th];

                    for (int inch = 0; inch < inchannels; inch++) {
                        x[inch, th] += v * w[inch, outch];
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 8, outchannels = 12, batch = 3;

            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            QuaternionMap0D y = new QuaternionMap0D(outchannels / 4, batch, ycval);
            QuaternionFilter0D w = new QuaternionFilter0D(inchannels / 4, outchannels / 4, wcval);

            QuaternionMap0D x = Reference(y, w);

            float[] x_expect = {
                -3.880000e-04f,  2.080000e-04f,  4.120000e-04f,  2.740000e-04f,  -2.200000e-04f,  1.120000e-04f,
                2.680000e-04f,  1.540000e-04f,  -1.252000e-03f,  1.216000e-03f,  1.492000e-03f,  1.210000e-03f,
                -7.960000e-04f,  8.320000e-04f,  1.060000e-03f,  8.020000e-04f,  -2.116000e-03f,  2.224000e-03f,
                2.572000e-03f,  2.146000e-03f,  -1.372000e-03f,  1.552000e-03f,  1.852000e-03f,  1.450000e-03f,
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{batch}");
        }
    }
}
