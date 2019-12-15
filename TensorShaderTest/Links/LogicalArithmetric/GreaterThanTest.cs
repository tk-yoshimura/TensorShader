using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.LogicalArithmetric {
    [TestClass]
    public class GreaterThanTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256, ch = 8;
            Random rd = new Random(1234);

            int[] idxes = (new int[length * ch]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length * ch]).Select((_) => (float)rd.Next(5)).ToArray();
            float[] x2 = (new float[length * ch]).Select((_) => (float)rd.Next(5)).ToArray();

            VariableField f1 = new Tensor(Shape.Map1D(ch, length), x1);
            VariableField f2 = new Tensor(Shape.Map1D(ch, length), x2);

            Field fout = GreaterThan(f1, f2);
            StoreField output = fout.Save();

            Flow flow = Flow.Inference(output);
            flow.Execute();

            CollectionAssert.AreEqual(idxes.Select((idx) => x1[idx] > x2[idx] ? 1f : 0f).ToArray(), output.State);
        }
    }
}
