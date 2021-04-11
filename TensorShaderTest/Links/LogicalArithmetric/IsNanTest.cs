using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.LogicalArithmetric {
    [TestClass]
    public class IsNanTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256, ch = 8;
            Random rd = new(1234);

            int[] idxes = (new int[length * ch]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length * ch]).Select((_) => (float)rd.Next(-2, 3) / rd.Next(-2, 3)).ToArray();

            VariableField f = (Shape.Map1D(ch, length), x);

            StoreField output = IsNan(f);

            (Flow flow, _) = Flow.Inference(output);
            flow.Execute();

            CollectionAssert.AreEqual(idxes.Select((idx) => float.IsNaN(x[idx]) ? 1f : 0f).ToArray(), output.State.Value);
        }
    }
}
