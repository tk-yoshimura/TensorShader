using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest.Functions.UnaryArithmetric {
    [TestClass]
    public class SignedSqrtTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length]).Select((_) => (float)rd.NextDouble() * 2f - 1f).ToArray();

            {
                Tensor t = x;

                Tensor o = Tensor.SignedSqrt(t);

                AssertError.Tolerance(idxes.Select((idx) => Math.Sign(x[idx]) * (float)Math.Sqrt(Math.Abs(x[idx]))).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t = x;

                var n = t + 0;

                OutputNode o = VariableNode.SignedSqrt(n).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => Math.Sign(x[idx]) * (float)Math.Sqrt(Math.Abs(x[idx]))).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }
        }
    }
}
