using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest.Functions.RandomGeneration {
    [TestClass]
    public class NormalRandomTest {
        [TestMethod]
        public void ExecuteTest() {
            {
                int channels = 32, width = 65536, length = channels * width;

                Shape shape = Shape.Map0D(channels, width);
                Random rd = new(1234);

                Tensor t = Tensor.NormalRandom(shape, rd);

                Assert.AreEqual(shape, t.Shape);

                float[] y = t.State.Value;

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
                int length = 5000;

                Shape shape = Shape.Vector(length);

                InputNode node = VariableNode.NormalRandom(shape, new Random(1234));
                OutputNode output = node.Save();

                Flow flow = Flow.FromInputs(node);

                flow.Execute();
                float[] y1 = output.State.Value;

                flow.Execute();
                float[] y2 = output.State.Value;

                CollectionAssert.AreNotEqual(y1, y2);

                Assert.IsTrue(y2.Min() < -1);
                Assert.IsTrue(y2.Max() > 1);
            }
        }
    }
}
