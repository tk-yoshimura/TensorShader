using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.UnaryArithmetric {
    [TestClass]
    public class MinimumTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length]).Select((_) => (float)rd.NextDouble() - 0.5f).ToArray();
            float c = 0;

            {
                Tensor t = new Tensor(Shape.Vector(length), x);

                Tensor o = Tensor.Minimum(t, c);

                AssertError.Tolerance(idxes.Select((idx) => Math.Min(x[idx], c)).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t = new Tensor(Shape.Vector(length), x);

                var n = t + 0;

                OutputNode o = VariableNode.Minimum(n, c).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => Math.Min(x[idx], c)).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                Tensor t = new Tensor(Shape.Vector(length), x);

                Tensor o = Tensor.Minimum(c, t);

                AssertError.Tolerance(idxes.Select((idx) => Math.Min(c, x[idx])).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t = new Tensor(Shape.Vector(length), x);

                var n = t + 0;

                OutputNode o = VariableNode.Minimum(c, n).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => Math.Min(c, x[idx])).ToArray(), o.State, 1e-7f, 1e-5f);
            }
        }
    }
}
