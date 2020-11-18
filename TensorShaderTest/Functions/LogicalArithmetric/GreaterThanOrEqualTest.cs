using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Tensor;

namespace TensorShaderTest.Functions.LogicalArithmetric {
    [TestClass]
    public class GreaterThanOrEqualTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256, ch = 8;
            Random rd = new Random(1234);

            int[] idxes = (new int[length * ch]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length * ch]).Select((_) => (float)rd.Next(5)).ToArray();
            float[] x2 = (new float[length * ch]).Select((_) => (float)rd.Next(5)).ToArray();

            {
                Tensor t1 = (Shape.Map1D(ch, length), x1);
                Tensor t2 = (Shape.Map1D(ch, length), x2);

                Tensor o = GreaterThanOrEqual(t1, t2);

                AssertError.Tolerance(idxes.Select((idx) => x1[idx] >= x2[idx] ? 1f : 0f).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                Tensor t1 = (Shape.Map1D(ch, length), x1);

                Tensor o = GreaterThanOrEqual(t1, t1);

                AssertError.Tolerance(idxes.Select((idx) => x1[idx] >= x1[idx] ? 1f : 0f).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = (Shape.Map1D(ch, length), x1);
                InputNode t2 = (Shape.Map1D(ch, length), x2);

                var n1 = t1 + 0;
                var n2 = t2 + 0;

                OutputNode o = VariableNode.GreaterThanOrEqual(n1, n2).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => x1[idx] >= x2[idx] ? 1f : 0f).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = (Shape.Map1D(ch, length), x1);
                InputNode t2 = (Shape.Map1D(ch, length), x2);

                var n1 = t1 + 0;

                OutputNode o = VariableNode.GreaterThanOrEqual(n1, t2).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => x1[idx] >= x2[idx] ? 1f : 0f).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = (Shape.Map1D(ch, length), x1);
                InputNode t2 = (Shape.Map1D(ch, length), x2);

                var n2 = t2 + 0;

                OutputNode o = VariableNode.GreaterThanOrEqual(t1, n2).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => x1[idx] >= x2[idx] ? 1f : 0f).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = (Shape.Map1D(ch, length), x1);

                var n1 = t1 + 0;

                OutputNode o = VariableNode.GreaterThanOrEqual(n1, n1).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => x1[idx] >= x1[idx] ? 1f : 0f).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }
        }
    }
}
