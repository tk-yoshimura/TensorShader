using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest.Functions.UnaryArithmetric {
    [TestClass]
    public class StepTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length]).Select((_) => (float)rd.Next(5) - 2).ToArray();

            {
                Tensor t = x;

                Tensor o = Tensor.Step(t);

                AssertError.Tolerance(idxes.Select((idx) => x[idx] >= 0 ? 1f : 0f).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t = x;

                var n = t + 0;

                OutputNode o = VariableNode.Step(n).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => x[idx] >= 0 ? 1f : 0f).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }
        }
    }
}
