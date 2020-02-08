using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;
using static TensorShader.VariableNode;

namespace TensorShaderTest.Links.Loss {
    [TestClass]
    public class HuberLossTest {
        [TestMethod]
        public void ReferenceTest() {
            float delta = 1.25f;

            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => idx * idx / 144f - 0.5f).ToArray();
            float[] tval = (new float[length]).Select((_, idx) => ((float)idx - 12) / 12).ToArray();

            ParameterField x = new Tensor(Shape.Vector(length), xval);
            VariableField t = new Tensor(Shape.Vector(length), tval);

            StoreField loss = HuberLoss(x, t, delta);

            StoreField diff = Abs(x - t);

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);

            flow.Execute();

            float[] diffval = diff.State;

            Assert.IsTrue(diffval.Where((v) => v > delta).Count() > 0);
            Assert.IsTrue(diffval.Where((v) => v < delta).Count() > 0);

            float[] loss_actual = loss.State;

            AssertError.Tolerance(loss_expect, loss_actual, 1e-7f, 1e-5f, $"not equal loss");

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] loss_expect = {
            1.25000000e-01f,
            8.97231996e-02f,
            6.52006119e-02f,
            4.88281250e-02f,
            3.85802500e-02f,
            3.30102183e-02f,
            3.12500000e-02f,
            3.30102183e-02f,
            3.85802574e-02f,
            4.88281250e-02f,
            6.52006119e-02f,
            8.97231996e-02f,
            1.25000000e-01f,
            1.74213976e-01f,
            2.41126478e-01f,
            3.30078125e-01f,
            4.45987731e-01f,
            5.94352841e-01f,
            7.81250000e-01f,
            9.98264313e-01f,
            1.23263860e+00f,
            1.48437500e+00f,
            1.75347233e+00f,
            2.03993034e+00f,
        };

        float[] gx_expect = {
            6.25000000e-02f,
            3.80077474e-02f,
            2.35446654e-02f,
            1.52587891e-02f,
            1.07167363e-02f,
            8.48179124e-03f,
            7.81250000e-03f,
            8.48179124e-03f,
            1.07167400e-02f,
            1.52587891e-02f,
            2.35446654e-02f,
            3.80077474e-02f,
            6.25000000e-02f,
            1.02834649e-01f,
            1.67448923e-01f,
            2.68188477e-01f,
            4.21210676e-01f,
            6.48009717e-01f,
            9.76562500e-01f,
            1.24783039e+00f,
            1.54079819e+00f,
            1.85546875e+00f,
            2.19184041e+00f,
            2.54991293e+00f,
        };
    }
}
