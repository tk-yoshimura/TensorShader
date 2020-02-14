using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.TrinaryArithmetric {
    [TestClass]
    public class ClampTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 1024;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();
            float[] xmin = (new float[length]).Select((_) => (float)rd.NextDouble() * 0.5f - 1).ToArray();
            float[] xmax = (new float[length]).Select((_) => (float)rd.NextDouble() * 0.5f + 0.5f).ToArray();

            {
                Tensor t1 = new Tensor(Shape.Vector(length), x);
                Tensor t2 = new Tensor(Shape.Vector(length), xmin);
                Tensor t3 = new Tensor(Shape.Vector(length), xmax);
                Tensor o = Tensor.Clamp(t1, t2, t3);

                AssertError.Tolerance(idxes.Select((idx) => Math.Min(Math.Max(x[idx], xmin[idx]), xmax[idx])).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = new Tensor(Shape.Vector(length), x);
                InputNode t2 = new Tensor(Shape.Vector(length), xmin);
                InputNode t3 = new Tensor(Shape.Vector(length), xmax);

                var n1 = t1 + 0;
                var n2 = t2 + 0;
                var n3 = t3 + 0;

                OutputNode o = VariableNode.Clamp(n1, n2, n3).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => Math.Min(Math.Max(x[idx], xmin[idx]), xmax[idx])).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                Tensor t1 = new Tensor(Shape.Vector(length), x);
                Tensor o = Tensor.Clamp(t1, -0.5f, 0.5f);

                AssertError.Tolerance(idxes.Select((idx) => Math.Min(Math.Max(x[idx], -0.5f), 0.5f)).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = new Tensor(Shape.Vector(length), x);

                var n1 = t1 + 0;

                OutputNode o = VariableNode.Clamp(n1, -0.5f, 0.5f).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => Math.Min(Math.Max(x[idx], -0.5f), 0.5f)).ToArray(), o.State, 1e-7f, 1e-5f);
            }
        }
    }
}
