using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.RandomGeneration;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.RandomGeneration {
    [TestClass]
    public class UniformRandomTest {
        [TestMethod]
        public void ExecuteTest() {
            int length = 65535 * 33;

            Shape shape = Shape.Vector(length);

            OverflowCheckedTensor v1 = new OverflowCheckedTensor(shape);

            UniformRandom ope = new UniformRandom(shape, new Random(1234));

            ope.Execute(v1);

            float[] y = v1.State;

            Assert.IsTrue(y.Min() >= 0);
            Assert.IsTrue(y.Max() < 1);

            {
                int[] cnt = new int[10];

                for (int i = 0; i < length; i++) {
                    cnt[(int)(y[i] * 10)]++;
                }

                for (int i = 0; i < 10; i++) {
                    Assert.AreEqual(0.1, (double)cnt[i] / length, 2e-3, $"[ {i * 0.1f}, {(i + 1) * 0.1f} )");
                }
            }

            {
                int[] cnt = new int[20];

                for (int i = 1; i < length; i++) {
                    float dy = y[i - 1] - y[i];

                    cnt[(int)Math.Floor(dy * 10 + 10)]++;
                }

                Assert.AreEqual(0.005, (double)cnt[0] / (length - 1), 5e-3);
                Assert.AreEqual(0.015, (double)cnt[1] / (length - 1), 5e-3);
                Assert.AreEqual(0.025, (double)cnt[2] / (length - 1), 5e-3);
                Assert.AreEqual(0.035, (double)cnt[3] / (length - 1), 5e-3);
                Assert.AreEqual(0.045, (double)cnt[4] / (length - 1), 5e-3);
                Assert.AreEqual(0.055, (double)cnt[5] / (length - 1), 5e-3);
                Assert.AreEqual(0.065, (double)cnt[6] / (length - 1), 5e-3);
                Assert.AreEqual(0.075, (double)cnt[7] / (length - 1), 5e-3);
                Assert.AreEqual(0.085, (double)cnt[8] / (length - 1), 5e-3);
                Assert.AreEqual(0.095, (double)cnt[9] / (length - 1), 5e-3);
                Assert.AreEqual(0.095, (double)cnt[10] / (length - 1), 5e-3);
                Assert.AreEqual(0.085, (double)cnt[11] / (length - 1), 5e-3);
                Assert.AreEqual(0.075, (double)cnt[12] / (length - 1), 5e-3);
                Assert.AreEqual(0.065, (double)cnt[13] / (length - 1), 5e-3);
                Assert.AreEqual(0.055, (double)cnt[14] / (length - 1), 5e-3);
                Assert.AreEqual(0.045, (double)cnt[15] / (length - 1), 5e-3);
                Assert.AreEqual(0.035, (double)cnt[16] / (length - 1), 5e-3);
                Assert.AreEqual(0.025, (double)cnt[17] / (length - 1), 5e-3);
                Assert.AreEqual(0.015, (double)cnt[18] / (length - 1), 5e-3);
                Assert.AreEqual(0.005, (double)cnt[19] / (length - 1), 5e-3);
            }

            {
                int[] cnt = new int[20];

                int sft = (int)TensorShaderCudaBackend.Shaders.Randomize.Randomize.RandomPerWarp;

                for (int i = sft; i < length; i++) {
                    float dy = y[i - sft] - y[i];

                    cnt[(int)Math.Floor(dy * 10 + 10)]++;
                }

                Assert.AreEqual(0.005, (double)cnt[0] / (length - sft), 5e-3);
                Assert.AreEqual(0.015, (double)cnt[1] / (length - sft), 5e-3);
                Assert.AreEqual(0.025, (double)cnt[2] / (length - sft), 5e-3);
                Assert.AreEqual(0.035, (double)cnt[3] / (length - sft), 5e-3);
                Assert.AreEqual(0.045, (double)cnt[4] / (length - sft), 5e-3);
                Assert.AreEqual(0.055, (double)cnt[5] / (length - sft), 5e-3);
                Assert.AreEqual(0.065, (double)cnt[6] / (length - sft), 5e-3);
                Assert.AreEqual(0.075, (double)cnt[7] / (length - sft), 5e-3);
                Assert.AreEqual(0.085, (double)cnt[8] / (length - sft), 5e-3);
                Assert.AreEqual(0.095, (double)cnt[9] / (length - sft), 5e-3);
                Assert.AreEqual(0.095, (double)cnt[10] / (length - sft), 5e-3);
                Assert.AreEqual(0.085, (double)cnt[11] / (length - sft), 5e-3);
                Assert.AreEqual(0.075, (double)cnt[12] / (length - sft), 5e-3);
                Assert.AreEqual(0.065, (double)cnt[13] / (length - sft), 5e-3);
                Assert.AreEqual(0.055, (double)cnt[14] / (length - sft), 5e-3);
                Assert.AreEqual(0.045, (double)cnt[15] / (length - sft), 5e-3);
                Assert.AreEqual(0.035, (double)cnt[16] / (length - sft), 5e-3);
                Assert.AreEqual(0.025, (double)cnt[17] / (length - sft), 5e-3);
                Assert.AreEqual(0.015, (double)cnt[18] / (length - sft), 5e-3);
                Assert.AreEqual(0.005, (double)cnt[19] / (length - sft), 5e-3);
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 65535 * 16;

            Shape shape = Shape.Vector(length);

            OverflowCheckedTensor v1 = new OverflowCheckedTensor(shape);

            UniformRandom ope = new UniformRandom(shape, new Random(1234));

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/uniform_random.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(v1);

            Cuda.Profiler.Stop();
        }
    }
}
