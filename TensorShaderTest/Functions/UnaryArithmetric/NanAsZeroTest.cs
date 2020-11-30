using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest.Functions.UnaryArithmetric {
    [TestClass]
    public class NanAsZeroTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length]).Select((_) => (float)rd.Next(-2, 3) / rd.Next(-2, 3)).ToArray();

            {
                Tensor t = x;

                Tensor o = Tensor.NanAsZero(t);

                AssertError.Tolerance(idxes.Select((idx) => float.IsNaN(x[idx]) ? 0 : x[idx]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t = x;

                var n = t + 0;

                OutputNode o = VariableNode.NanAsZero(n).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => float.IsNaN(x[idx]) ? 0 : x[idx]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }
        }
    }
}
