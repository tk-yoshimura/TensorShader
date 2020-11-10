using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.TrinaryArithmetric {
    [TestClass]
    public class CustomTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();
            float[] x2 = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();
            float[] x3 = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();

            {
                Tensor t1 = (Shape.Vector(length), x1);
                Tensor t2 = (Shape.Vector(length), x2);
                Tensor t3 = (Shape.Vector(length), x3);
                Tensor o = Tensor.TrinaryArithmetric(t1, t2, t3, "trinary_test", "#y = (#x1 + #x2) * (#x1 - #x3);");

                AssertError.Tolerance(idxes.Select((idx) => (x1[idx] + x2[idx]) * (x1[idx] - x3[idx])).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = (Shape.Vector(length), x1);
                InputNode t2 = (Shape.Vector(length), x2);
                InputNode t3 = (Shape.Vector(length), x3);

                var n1 = t1 + 0;
                var n2 = t2 + 0;
                var n3 = t3 + 0;

                OutputNode o = VariableNode.TrinaryArithmetric(t1, t2, t3, "trinary_test", "#y = (#x1 + #x2) * (#x1 - #x3);").Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => (x1[idx] + x2[idx]) * (x1[idx] - x3[idx])).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                Tensor t1 = (Shape.Vector(length), x1);
                Tensor t2 = (Shape.Vector(length), x2);
                Tensor o = Tensor.TrinaryUniConstantArithmetric(4, t1, t2, "trinaryuniconst_test", "#y = c * (#x1 + #x2);");

                AssertError.Tolerance(idxes.Select((idx) => 4 * (x1[idx] + x2[idx])).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = (Shape.Vector(length), x1);
                InputNode t2 = (Shape.Vector(length), x2);

                var n1 = t1 + 0;
                var n2 = t2 + 0;

                OutputNode o = VariableNode.TrinaryUniConstantArithmetric(4, n1, n2, "trinaryuniconst_test", "#y = c * (#x1 + #x2);").Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => 4 * (x1[idx] + x2[idx])).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                Tensor t1 = (Shape.Vector(length), x1);
                Tensor o = Tensor.TrinaryBiConstantArithmetric(4, 5, t1, "trinarybiconst_test", "#y = c1 * (#x + c2);");

                AssertError.Tolerance(idxes.Select((idx) => 4 * (x1[idx] + 5)).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = (Shape.Vector(length), x1);

                var n1 = t1 + 0;

                OutputNode o = VariableNode.TrinaryBiConstantArithmetric(4, 5, n1, "trinarybiconst_test", "#y = c * (#x + c2);").Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => 4 * (x1[idx] + 5)).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                Tensor t1 = (Shape.Vector(length), x1);
                Tensor t2 = (Shape.Vector(length), x2);
                Tensor o = Tensor.TrinaryUniConstantArithmetric(8, t1, t2, "trinaryuniconst_test", "#y = c * (#x1 + #x2);");

                AssertError.Tolerance(idxes.Select((idx) => 8 * (x1[idx] + x2[idx])).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = (Shape.Vector(length), x1);
                InputNode t2 = (Shape.Vector(length), x2);

                var n1 = t1 + 0;
                var n2 = t2 + 0;

                OutputNode o = VariableNode.TrinaryUniConstantArithmetric(8, n1, n2, "trinaryuniconst_test", "#y = c * (#x1 + #x2);").Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => 8 * (x1[idx] + x2[idx])).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                Tensor t1 = (Shape.Vector(length), x1);
                Tensor o = Tensor.TrinaryBiConstantArithmetric(8, 10, t1, "trinarybiconst_test", "#y = c1 * (#x + c2);");

                AssertError.Tolerance(idxes.Select((idx) => 8 * (x1[idx] + 10)).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = (Shape.Vector(length), x1);

                var n1 = t1 + 0;

                OutputNode o = VariableNode.TrinaryBiConstantArithmetric(8, 10, n1, "trinarybiconst_test", "#y = c * (#x + c2);").Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => 8 * (x1[idx] + 10)).ToArray(), o.State, 1e-7f, 1e-5f);
            }
        }
    }
}
