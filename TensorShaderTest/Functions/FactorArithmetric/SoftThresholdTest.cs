using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.FactorArithmetric {
    [TestClass]
    public class SoftThresholdTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256, ch = 8;
            Random rd = new Random(1234);

            int[] idxes = (new int[length * ch]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length * ch]).Select((_) => (float)rd.NextDouble() * 8 - 4).ToArray();
            float[] x2 = { (float)rd.NextDouble() * 8 - 4 };

            float[] coef = { (float)rd.NextDouble() };

            Func<float, float, float> soft_thr = (x, c) => {
                return Math.Sign(x) * Math.Max(Math.Abs(x) - c, 0);
            };

            {
                Tensor t1 = new Tensor(Shape.Map1D(ch, length), x1);
                Tensor t2 = new Tensor(Shape.Scalar(), coef);

                Tensor o = Tensor.SoftThreshold(t1, t2);

                AssertError.Tolerance(idxes.Select((idx) => soft_thr(x1[idx], coef[0])).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                Tensor t1 = new Tensor(Shape.Scalar(), x2);
                Tensor t2 = new Tensor(Shape.Scalar(), coef);

                Tensor o = Tensor.SoftThreshold(t1, t2);

                AssertError.Tolerance(new float[] { soft_thr(x2[0], coef[0]) }, o.State, 1e-7f, 1e-5f);
            }

            {
                Tensor t1 = new Tensor(Shape.Scalar(), x2);

                Tensor o = Tensor.SoftThreshold(t1, t1);

                AssertError.Tolerance(x2.Select((v) => soft_thr(v, v)).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = new Tensor(Shape.Map1D(ch, length), x1);
                InputNode t2 = new Tensor(Shape.Scalar(), coef);

                var n1 = t1 + 0;
                var n2 = t2 + 0;

                OutputNode o = VariableNode.SoftThreshold(n1, n2).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => soft_thr(x1[idx], coef[0])).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = new Tensor(Shape.Scalar(), x2);
                InputNode t2 = new Tensor(Shape.Scalar(), coef);

                var n1 = t1 + 0;
                var n2 = t2 + 0;

                OutputNode o = VariableNode.SoftThreshold(n1, n2).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(new float[] { soft_thr(x2[0], coef[0]) }, o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t1 = new Tensor(Shape.Scalar(), x2);

                var n1 = t1 + 0;

                OutputNode o = VariableNode.SoftThreshold(n1, n1).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(x2.Select((v) => soft_thr(v, v)).ToArray(), o.State, 1e-7f, 1e-5f);
            }
        }
    }
}
