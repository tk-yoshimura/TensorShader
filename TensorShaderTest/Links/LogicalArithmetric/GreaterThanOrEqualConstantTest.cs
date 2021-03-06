using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.LogicalArithmetric {
    [TestClass]
    public class GreaterThanOrEqualConstantTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256, ch = 8;
            Random rd = new(1234);

            int[] idxes = (new int[length * ch]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length * ch]).Select((_) => (float)rd.Next(5)).ToArray();

            VariableField f = (Shape.Map1D(ch, length), x);

            StoreField output = GreaterThanOrEqual(f, 2);

            (Flow flow, _) = Flow.Inference(output);
            flow.Execute();

            CollectionAssert.AreEqual(idxes.Select((idx) => x[idx] >= 2 ? 1f : 0f).ToArray(), output.State.Value);
        }
    }
}
