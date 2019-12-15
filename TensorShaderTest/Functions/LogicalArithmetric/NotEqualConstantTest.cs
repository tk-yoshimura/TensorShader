using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.LogicalArithmetric {
    [TestClass]
    public class NotEqualConstantTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length]).Select((_) => (float)rd.Next(5)).ToArray();

            {
                Tensor t = new Tensor(Shape.Vector(length), x);

                Tensor o = Tensor.NotEqual(t, 2);

                AssertError.Tolerance(idxes.Select((idx) => x[idx] != 2 ? 1f : 0f).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t = new Tensor(Shape.Vector(length), x);

                var n = t + 0;

                OutputNode o = VariableNode.NotEqual(n, 2).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => x[idx] != 2 ? 1f : 0f).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                Tensor t = new Tensor(Shape.Vector(length), x);

                Tensor o = Tensor.NotEqual(2, t);

                AssertError.Tolerance(idxes.Select((idx) => 2 != x[idx] ? 1f : 0f).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t = new Tensor(Shape.Vector(length), x);

                var n = t + 0;

                OutputNode o = VariableNode.NotEqual(2, n).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => 2 != x[idx] ? 1f : 0f).ToArray(), o.State, 1e-7f, 1e-5f);
            }
        }
    }
}
