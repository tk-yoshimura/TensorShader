using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest.Functions.BinaryArithmetric {
    [TestClass]
    public class DivTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256, ch = 8;
            Random rd = new Random(1234);

            int[] idxes = (new int[length * ch]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length * ch]).Select((_) => (float)rd.NextDouble() + 1f).ToArray();
            float[] x2 = (new float[length * ch]).Select((_) => (float)rd.NextDouble() + 1f).ToArray();
            float[] x3 = (new float[ch]).Select((_) => (float)rd.NextDouble() + 1f).ToArray();
            float[] x4 = { (float)rd.NextDouble() + 1f };

            {
                Tensor t1 = (Shape.Map1D(ch, length), x1);
                Tensor t2 = (Shape.Map1D(ch, length), x2);

                Tensor o = t1 / t2;

                AssertError.Tolerance(idxes.Select((idx) => x1[idx] / x2[idx]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                Tensor t1 = (Shape.Map1D(ch, length), x1);

                Tensor o = t1 / t1;

                AssertError.Tolerance(idxes.Select((idx) => x1[idx] / x1[idx]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                Tensor t1 = (Shape.Map1D(ch, length), x1);
                Tensor t3 = (Shape.Vector(ch), x3);

                Tensor o = t1 / t3;

                AssertError.Tolerance(idxes.Select((idx) => x1[idx] / x3[idx % ch]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                Tensor t1 = (Shape.Map1D(ch, length), x1);
                Tensor t3 = (Shape.Vector(ch), x3);

                Tensor o = t3 / t1;

                AssertError.Tolerance(idxes.Select((idx) => x3[idx % ch] / x1[idx]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                Tensor t1 = (Shape.Map1D(ch, length), x1);
                Tensor t4 = (Shape.Scalar, x4);

                Tensor o = t1 / t4;

                AssertError.Tolerance(idxes.Select((idx) => x1[idx] / x4[0]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                Tensor t1 = (Shape.Map1D(ch, length), x1);
                Tensor t4 = (Shape.Scalar, x4);

                Tensor o = t4 / t1;

                AssertError.Tolerance(idxes.Select((idx) => x4[0] / x1[idx]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = (Shape.Map1D(ch, length), x1);
                InputNode t2 = (Shape.Map1D(ch, length), x2);

                var n1 = t1 + 0;
                var n2 = t2 + 0;

                OutputNode o = (n1 / n2).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => x1[idx] / x2[idx]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = (Shape.Map1D(ch, length), x1);
                InputNode t2 = (Shape.Map1D(ch, length), x2);

                var n1 = t1 + 0;

                OutputNode o = (n1 / t2).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => x1[idx] / x2[idx]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = (Shape.Map1D(ch, length), x1);
                InputNode t2 = (Shape.Map1D(ch, length), x2);

                var n2 = t2 + 0;

                OutputNode o = (t1 / n2).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => x1[idx] / x2[idx]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = (Shape.Map1D(ch, length), x1);

                var n1 = t1 + 0;

                OutputNode o = (n1 / n1).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => x1[idx] / x1[idx]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = (Shape.Map1D(ch, length), x1);
                InputNode t3 = (Shape.Vector(ch), x3);

                var n1 = t1 + 0;
                var n3 = t3 + 0;

                OutputNode o = (n1 / n3).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => x1[idx] / x3[idx % ch]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = (Shape.Map1D(ch, length), x1);
                InputNode t3 = (Shape.Vector(ch), x3);

                var n1 = t1 + 0;
                var n3 = t3 + 0;

                OutputNode o = (n3 / n1).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => x3[idx % ch] / x1[idx]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = (Shape.Map1D(ch, length), x1);
                InputNode t4 = (Shape.Scalar, x4);

                var n1 = t1 + 0;
                var n4 = t4 + 0;

                OutputNode o = (n1 / n4).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => x1[idx] / x4[0]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = (Shape.Map1D(ch, length), x1);
                InputNode t4 = (Shape.Scalar, x4);

                var n1 = t1 + 0;
                var n4 = t4 + 0;

                OutputNode o = (n4 / n1).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => x4[0] / x1[idx]).ToArray(), o.State.Value, 1e-7f, 1e-5f);
            }
        }
    }
}
