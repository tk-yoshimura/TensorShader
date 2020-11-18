using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.BinaryArithmetric {
    [TestClass]
    public class LimitAbsTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();
            float[] x2 = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();

            {
                Tensor t1 = x1;
                Tensor t2 = x2;
                Tensor o = Tensor.LimitAbs(t1, t2);

                AssertError.Tolerance(idxes.Select((idx) => Math.Min(Math.Max(x1[idx], -x2[idx]), x2[idx])).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                Tensor t1 = x1;

                Tensor o = Tensor.LimitAbs(t1, t1);

                AssertError.Tolerance(idxes.Select((idx) => Math.Min(Math.Max(x1[idx], -x1[idx]), x1[idx])).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = x1;
                InputNode t2 = x2;

                var n1 = t1 + 0;
                var n2 = t2 + 0;

                OutputNode o = VariableNode.LimitAbs(n1, n2).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => Math.Min(Math.Max(x1[idx], -x2[idx]), x2[idx])).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = x1;
                InputNode t2 = x2;

                var n1 = t1 + 0;

                OutputNode o = VariableNode.LimitAbs(n1, t2).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => Math.Min(Math.Max(x1[idx], -x2[idx]), x2[idx])).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = x1;
                InputNode t2 = x2;

                var n2 = t2 + 0;

                OutputNode o = VariableNode.LimitAbs(t1, n2).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => Math.Min(Math.Max(x1[idx], -x2[idx]), x2[idx])).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = x1;

                var n1 = t1 + 0;

                OutputNode o = VariableNode.LimitAbs(n1, n1).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => Math.Min(Math.Max(x1[idx], -x1[idx]), x1[idx])).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }
        }
    }
}
