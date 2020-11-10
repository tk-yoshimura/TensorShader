using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.BinaryArithmetric {
    [TestClass]
    public class Arctan2Test {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();
            float[] x2 = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();

            {
                Tensor t1 = x1;
                Tensor t2 = x2;
                Tensor o = Tensor.Arctan2(t1, t2);

                AssertError.Tolerance(idxes.Select((idx) => (float)Math.Atan2(x1[idx], x2[idx])).ToArray(), o.State, 1e-7f, 1e-3f); /*non linear tolerance*/
            }

            {
                Tensor t1 = x1;

                Tensor o = Tensor.Arctan2(t1, t1);

                AssertError.Tolerance(idxes.Select((idx) => (float)Math.Atan2(x1[idx], x1[idx])).ToArray(), o.State, 1e-7f, 1e-3f); /*non linear tolerance*/
            }

            {
                InputNode t1 = x1;
                InputNode t2 = x2;

                var n1 = t1 + 0;
                var n2 = t2 + 0;

                OutputNode o = VariableNode.Arctan2(n1, n2).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => (float)Math.Atan2(x1[idx], x2[idx])).ToArray(), o.State, 1e-7f, 1e-3f); /*non linear tolerance*/
            }

            {
                InputNode t1 = x1;
                InputNode t2 = x2;

                var n1 = t1 + 0;

                OutputNode o = VariableNode.Arctan2(n1, t2).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => (float)Math.Atan2(x1[idx], x2[idx])).ToArray(), o.State, 1e-7f, 1e-3f); /*non linear tolerance*/
            }

            {
                InputNode t1 = x1;
                InputNode t2 = x2;

                var n2 = t2 + 0;

                OutputNode o = VariableNode.Arctan2(t1, n2).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => (float)Math.Atan2(x1[idx], x2[idx])).ToArray(), o.State, 1e-7f, 1e-3f); /*non linear tolerance*/
            }

            {
                InputNode t1 = x1;

                var n1 = t1 + 0;

                OutputNode o = VariableNode.Arctan2(n1, n1).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => (float)Math.Atan2(x1[idx], x1[idx])).ToArray(), o.State, 1e-7f, 1e-3f); /*non linear tolerance*/
            }
        }
    }
}
