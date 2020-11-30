using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.RandomGeneration {
    [TestClass]
    public class BinaryRandomTest {
        [TestMethod]
        public void ExecuteTest() {
            int length = 5000;

            Shape shape = Shape.Vector(length);

            StoreField output = BinaryRandom(shape, new Random(1234), 0.5f);
            (Flow flow, _) = Flow.Inference(output);

            flow.Execute();
            float[] y1 = output.State.Value;

            flow.Execute();
            float[] y2 = output.State.Value;

            CollectionAssert.AreNotEqual(y1, y2);

            Assert.AreEqual(0, y2.Where((v) => v > 1e-5f && v < 1 - 1e-5f).Count());
        }
    }
}
