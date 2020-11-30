using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest.Functions.RandomGeneration {
    [TestClass]
    public class UniformRandomTest {
        [TestMethod]
        public void ExecuteTest() {
            {
                int channels = 4, width = 65536, length = channels * width;

                Shape shape = Shape.Map0D(channels, width);
                Random rd = new Random(1234);

                Tensor t = Tensor.UniformRandom(shape, rd);

                Assert.AreEqual(shape, t.Shape);

                float[] y = t.State.Value;

                Assert.IsTrue(y.Min() >= 0);
                Assert.IsTrue(y.Max() < 1);

                int[] cnt = new int[length];

                for (int i = 0; i < length; i++) {
                    cnt[(int)(y[i] * 10)]++;
                }

                for (int i = 0; i < 10; i++) {
                    Assert.AreEqual(0.1, (double)cnt[i] / length, 2e-3, $"[ {i * 0.1f}, {(i + 1) * 0.1f} )");
                }
            }

            {
                int length = 5000;

                Shape shape = Shape.Vector(length);

                InputNode node = VariableNode.UniformRandom(shape, new Random(1234));
                OutputNode output = node.Save();

                Flow flow = Flow.FromInputs(node);

                flow.Execute();
                float[] y1 = output.State.Value;

                flow.Execute();
                float[] y2 = output.State.Value;

                CollectionAssert.AreNotEqual(y1, y2);

                Assert.IsTrue(y2.Min() >= 0);
                Assert.IsTrue(y2.Max() < 1);
            }
        }

    }
}
