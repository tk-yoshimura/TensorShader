using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.RandomGeneration;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.RandomGeneration {
    [TestClass]
    public class NormalRandomTest {
        [TestMethod]
        public void ExecuteTest() {
            int length = 65535 * 63;

            Shape shape = Shape.Vector(length);

            OverflowCheckedTensor v1 = new OverflowCheckedTensor(shape);

            NormalRandom ope = new NormalRandom(shape, new Random(1234));

            ope.Execute(v1);

            float[] y = v1.State.Value;

            {
                double sq_sum = 0, sum = 0;
                int[] cnt = new int[4];

                foreach (float v in y) {
                    sq_sum += v * v;
                    sum += v;

                    float abs_v = Math.Abs(v);

                    if (abs_v < 4) {
                        cnt[(int)Math.Floor(abs_v)]++;
                    }
                }

                double mean = sum / length;

                double variance = sq_sum / length - mean * mean;

                Assert.AreEqual(0, mean, 2e-3, "mean");
                Assert.AreEqual(1, variance, 2e-3, "variance");

                Assert.AreEqual(0.682689492, (double)cnt[0] / length, 1e-3, "sigma1");
                Assert.AreEqual(0.271810244, (double)cnt[1] / length, 1e-3, "sigma2");
                Assert.AreEqual(0.042800468, (double)cnt[2] / length, 1e-3, "sigma3");
                Assert.AreEqual(0.002636456, (double)cnt[3] / length, 1e-3, "sigma4");
            }

            {
                double sq_sum = 0, sum = 0;
                int[] cnt = new int[8];

                double sum_xy = 0;

                for (int i = 1; i < y.Length; i++) {
                    float v = y[i] - y[i - 1];

                    sum_xy += y[i] * y[i - 1];

                    sq_sum += v * v;
                    sum += v;

                    float abs_v = Math.Abs(v);

                    if (abs_v < 8) {
                        cnt[(int)Math.Floor(abs_v)]++;
                    }
                }

                double mean = sum / length;

                double variance = sq_sum / length - mean * mean;

                Assert.AreEqual(0, mean, 1e-2, "mean");
                Assert.AreEqual(2, variance, 1e-2, "variance");
                Assert.AreEqual(0, sum_xy / (length - 1), 1e-2, "cov");

                Assert.AreEqual(0.520499878, (double)cnt[0] / (length - 1), 2e-2, "sigma1");
                Assert.AreEqual(0.322200915, (double)cnt[1] / (length - 1), 2e-2, "sigma2");
                Assert.AreEqual(0.123404354, (double)cnt[2] / (length - 1), 2e-2, "sigma3");
                Assert.AreEqual(0.029217119, (double)cnt[3] / (length - 1), 2e-2, "sigma4");
                Assert.AreEqual(0.004270783, (double)cnt[4] / (length - 1), 2e-2, "sigma5");
                Assert.AreEqual(0.000384861, (double)cnt[5] / (length - 1), 2e-2, "sigma6");
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 65535 * 16;

            Shape shape = Shape.Vector(length);

            OverflowCheckedTensor v1 = new OverflowCheckedTensor(shape);

            NormalRandom ope = new NormalRandom(shape, new Random(1234));

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/normal_random.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(v1);

            Cuda.Profiler.Stop();
        }
    }
}
