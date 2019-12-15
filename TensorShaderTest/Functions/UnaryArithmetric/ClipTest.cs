using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.UnaryArithmetric {
    [TestClass]
    public class ClipTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length]).Select((_) => (float)rd.NextDouble() - 0.5f).ToArray();
            float cmin = -0.2f, cmax = 0.3f;

            {
                Tensor t = new Tensor(Shape.Vector(length), x);

                Tensor o = Tensor.Clip(t, cmin, cmax);

                AssertError.Tolerance(idxes.Select((idx) => Math.Min(Math.Max(x[idx], cmin), cmax)).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t = new Tensor(Shape.Vector(length), x);

                var n = t + 0;

                OutputNode o = VariableNode.Clip(n, cmin, cmax).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => Math.Min(Math.Max(x[idx], cmin), cmax)).ToArray(), o.State, 1e-7f, 1e-5f);
            }
        }
    }
}
