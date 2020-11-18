using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.UnaryArithmetric {
    [TestClass]
    public class PowTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length]).Select((_) => (float)rd.NextDouble() + 1f).ToArray();

            {
                Tensor t = x;

                Tensor o = Tensor.Pow(t, 1.5f);

                AssertError.Tolerance(idxes.Select((idx) => (float)Math.Pow(x[idx], 1.5)).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t = x;

                var n = t + 0;

                OutputNode o = VariableNode.Pow(n, 1.5f).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => (float)Math.Pow(x[idx], 1.5)).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }
        }
    }
}
