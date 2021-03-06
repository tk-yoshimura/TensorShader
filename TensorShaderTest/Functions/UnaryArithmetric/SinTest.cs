using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest.Functions.UnaryArithmetric {
    [TestClass]
    public class SinTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length]).Select((_) => (float)rd.NextDouble() * 4).ToArray();

            {
                Tensor t = x;

                Tensor o = Tensor.Sin(t);

                AssertError.Tolerance(idxes.Select((idx) => (float)Math.Sin(x[idx])).ToArray(), o.State.Value, 1e-6f, 1e-4f); /*nonlinear tolerance*/
            }

            {
                InputNode t = x;

                var n = t + 0;

                OutputNode o = VariableNode.Sin(n).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => (float)Math.Sin(x[idx])).ToArray(), o.State.Value, 1e-6f, 1e-4f); /*nonlinear tolerance*/
            }
        }
    }
}
