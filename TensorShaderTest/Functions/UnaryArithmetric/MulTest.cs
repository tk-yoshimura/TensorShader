using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest.Functions.UnaryArithmetric {
    [TestClass]
    public class MulTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length]).Select((_) => (float)rd.NextDouble() - 0.5f).ToArray();
            float c = (float)rd.NextDouble();

            {
                Tensor t = x;

                Tensor o = t * c;

                AssertError.Tolerance(idxes.Select((idx) => x[idx] * c).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t = x;

                var n = t + 0;

                OutputNode o = (n * c).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => x[idx] * c).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                Tensor t = x;

                Tensor o = c * t;

                AssertError.Tolerance(idxes.Select((idx) => c * x[idx]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t = x;

                var n = t + 0;

                OutputNode o = (c * n).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => c * x[idx]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }
        }
    }
}
