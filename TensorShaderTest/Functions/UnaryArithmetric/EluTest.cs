using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.UnaryArithmetric {
    [TestClass]
    public class EluTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length]).Select((_) => (float)rd.NextDouble() * 4 - 2).ToArray();
            float alpha = 0.5f;

            float elu(float v) {
                return v > 0 ? v : alpha * ((float)Math.Exp(v) - 1);
            }

            {
                Tensor t = (Shape.Vector(length), x);

                Tensor o = Tensor.Elu(t, alpha);

                AssertError.Tolerance(idxes.Select((idx) => elu(x[idx])).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t = (Shape.Vector(length), x);

                var n = t + 0;

                OutputNode o = VariableNode.Elu(n, alpha).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => elu(x[idx])).ToArray(), o.State, 1e-7f, 1e-5f);
            }
        }
    }
}
