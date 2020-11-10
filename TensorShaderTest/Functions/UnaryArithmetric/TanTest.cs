using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.UnaryArithmetric {
    [TestClass]
    public class TanTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length]).Select((_) => (float)rd.NextDouble() - 0.5f).ToArray();

            {
                Tensor t = (Shape.Vector(length), x);

                Tensor o = Tensor.Tan(t);

                AssertError.Tolerance(idxes.Select((idx) => (float)Math.Tan(x[idx])).ToArray(), o.State, 1e-6f, 1e-4f); /*nonlinear tolerance*/
            }

            {
                InputNode t = (Shape.Vector(length), x);

                var n = t + 0;

                OutputNode o = VariableNode.Tan(n).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => (float)Math.Tan(x[idx])).ToArray(), o.State, 1e-6f, 1e-4f); /*nonlinear tolerance*/
            }
        }
    }
}
