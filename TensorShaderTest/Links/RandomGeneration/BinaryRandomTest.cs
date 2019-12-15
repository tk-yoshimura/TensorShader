using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.RandomGeneration {
    [TestClass]
    public class BinaryRandomTest {
        [TestMethod]
        public void ExecuteTest() {
            int length = 5000;

            Shape shape = Shape.Vector(length);

            VariableField field = BinaryRandom(shape, new Random(1234), 0.5f);
            OutputNode output = field.Value.Save();
            Flow flow = Flow.Inference(output);

            flow.Execute();
            float[] y1 = output.Tensor.State;

            flow.Execute();
            float[] y2 = output.Tensor.State;

            CollectionAssert.AreNotEqual(y1, y2);

            Assert.AreEqual(0, y2.Where((v) => v > 1e-5f && v < 1 - 1e-5f).Count());
        }
    }
}
