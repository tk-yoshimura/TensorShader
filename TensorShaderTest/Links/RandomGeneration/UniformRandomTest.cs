using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.RandomGeneration {
    [TestClass]
    public class UniformRandomTest {
        [TestMethod]
        public void ExecuteTest() {
            int length = 5000;

            Shape shape = Shape.Vector(length);

            StoreField output = UniformRandom(shape, new Random(1234));
            (Flow flow, _) = Flow.Inference(output);

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
