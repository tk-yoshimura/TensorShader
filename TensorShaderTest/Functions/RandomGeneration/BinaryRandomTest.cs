using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest.Functions.RandomGeneration {
    [TestClass]
    public class BinaryRandomTest {
        [TestMethod]
        public void ExecuteTest() {
            for (decimal prob = 0; prob <= 1; prob += 0.05m) {
                int length = 65535 * 16;

                Shape shape = Shape.Vector(length);

                Tensor tensor = Tensor.BinaryRandom(shape, new Random(1234), (float)prob);

                float[] y = tensor.State.Value;

                Assert.AreEqual((float)prob, y.Average(), 1e-3f);
            }

            {
                int length = 5000;

                Shape shape = Shape.Vector(length);

                InputNode node = VariableNode.BinaryRandom(shape, new Random(1234), 0.5f);
                OutputNode output = node.Save();

                Flow flow = Flow.FromInputs(node);

                flow.Execute();
                float[] y1 = output.State.Value;

                flow.Execute();
                float[] y2 = output.State.Value;

                CollectionAssert.AreNotEqual(y1, y2);

                Assert.AreEqual(0, y2.Where((v) => v > 1e-5f && v < 1 - 1e-5f).Count());
            }
        }
    }
}
