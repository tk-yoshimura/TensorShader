using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest.Functions.UnaryArithmetric {
    [TestClass]
    public class LeakyReluTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length]).Select((_) => (float)rd.NextDouble() * 4 - 2).ToArray();
            float c = 0.25f;

            {
                Tensor t = x;

                Tensor o = Tensor.LeakyRelu(t, c);

                AssertError.Tolerance(idxes.Select((idx) => Math.Max(x[idx], 0) + c * Math.Min(x[idx], 0)).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t = x;

                var n = t + 0;

                OutputNode o = VariableNode.LeakyRelu(n, c).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => Math.Max(x[idx], 0) + c * Math.Min(x[idx], 0)).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }
        }
    }
}
