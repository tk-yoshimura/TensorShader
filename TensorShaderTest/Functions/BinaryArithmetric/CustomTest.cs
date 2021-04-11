using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest.Functions.BinaryArithmetric {
    [TestClass]
    public class CustomTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();
            float[] x2 = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();

            {
                Tensor t1 = x1;
                Tensor t2 = x2;
                Tensor o = Tensor.BinaryArithmetric(t1, t2, "binary_test", "#y = (#x1 + #x2) * (#x1 - #x2);");

                AssertError.Tolerance(idxes.Select((idx) => (x1[idx] + x2[idx]) * (x1[idx] - x2[idx])).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = x1;
                InputNode t2 = x2;

                var n1 = t1 + 0;
                var n2 = t2 + 0;

                OutputNode o = VariableNode.BinaryArithmetric(n1, n2, "binary_test", "#y = (#x1 + #x2) * (#x1 - #x2);").Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => (x1[idx] + x2[idx]) * (x1[idx] - x2[idx])).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                Tensor t1 = x1;
                Tensor o = Tensor.BinaryConstantArithmetric(4, t1, "binaryconst_test", "#y = c * (#x + c);");

                AssertError.Tolerance(idxes.Select((idx) => 4 * (x1[idx] + 4)).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = x1;

                var n1 = t1 + 0;

                OutputNode o = VariableNode.BinaryConstantArithmetric(4, n1, "binaryconst_test", "#y = c * (#x + c);").Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => 4 * (x1[idx] + 4)).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                Tensor t1 = x1;
                Tensor o = Tensor.BinaryConstantArithmetric(8, t1, "binaryconst_test", "#y = c * (#x + c);");

                AssertError.Tolerance(idxes.Select((idx) => 8 * (x1[idx] + 8)).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = x1;

                var n1 = t1 + 0;

                OutputNode o = VariableNode.BinaryConstantArithmetric(8, n1, "binaryconst_test", "#y = c * (#x + c);").Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => 8 * (x1[idx] + 8)).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }
        }
    }
}
