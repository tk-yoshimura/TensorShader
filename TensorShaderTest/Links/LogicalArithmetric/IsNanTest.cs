using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.LogicalArithmetric {
    [TestClass]
    public class IsNanTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256, ch = 8;
            Random rd = new Random(1234);

            int[] idxes = (new int[length * ch]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length * ch]).Select((_) => (float)rd.Next(-2, 3) / rd.Next(-2, 3)).ToArray();

            VariableField f = new Tensor(Shape.Map1D(ch, length), x);

            Field fout = IsNan(f);
            StoreField output = fout.Save();

            Flow flow = Flow.Inference(output);
            flow.Execute();

            CollectionAssert.AreEqual(idxes.Select((idx) => float.IsNaN(x[idx]) ? 1f : 0f).ToArray(), output.State);
        }
    }
}
