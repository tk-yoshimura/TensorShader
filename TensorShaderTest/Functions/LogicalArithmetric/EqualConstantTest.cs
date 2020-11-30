using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest.Functions.LogicalArithmetric {
    [TestClass]
    public class EqualConstantTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length]).Select((_) => (float)rd.Next(5)).ToArray();

            {
                Tensor t = x;

                Tensor o = Tensor.Equal(t, 2);

                AssertError.Tolerance(idxes.Select((idx) => x[idx] == 2 ? 1f : 0f).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t = x;

                var n = t + 0;

                OutputNode o = VariableNode.Equal(n, 2).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => x[idx] == 2 ? 1f : 0f).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                Tensor t = x;

                Tensor o = Tensor.Equal(2, t);

                AssertError.Tolerance(idxes.Select((idx) => 2 == x[idx] ? 1f : 0f).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t = x;

                var n = t + 0;

                OutputNode o = VariableNode.Equal(2, n).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => 2 == x[idx] ? 1f : 0f).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }
        }
    }
}
