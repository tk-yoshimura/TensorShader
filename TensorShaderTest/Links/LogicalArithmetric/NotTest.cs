using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.LogicalArithmetric {
    [TestClass]
    public class NotTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256, ch = 8;
            Random rd = new Random(1234);

            int[] idxes = (new int[length * ch]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length * ch]).Select((_) => (float)rd.Next(2)).ToArray();

            VariableField f1 = (Shape.Map1D(ch, length), x1);

            StoreField output = Not(f1);

            (Flow flow, _) = Flow.Inference(output);
            flow.Execute();

            CollectionAssert.AreEqual(idxes.Select((idx) => 1 - x1[idx]).ToArray(), output.State.Value);
        }
    }
}
