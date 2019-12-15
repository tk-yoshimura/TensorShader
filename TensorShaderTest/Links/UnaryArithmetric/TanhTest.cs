using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class TanhTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (idx + 0.5f) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = Tanh(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            4.15703582e-02f,
            8.14077010e-02f,
            1.16890160e-01f,
            1.45900452e-01f,
            1.67073803e-01f,
            1.79882460e-01f,
            1.84583108e-01f,
            1.82059882e-01f,
            1.73611548e-01f,
            1.60731404e-01f,
            1.44916898e-01f,
            1.27529495e-01f,
            1.09709859e-01f,
            9.23423740e-02f,
            7.60571787e-02f,
            6.12565503e-02f,
            4.81539127e-02f,
            3.68165663e-02f,
            2.72062608e-02f,
            1.92143462e-02f,
            1.26901898e-02f,
            7.46282251e-03f,
            3.35650954e-03f,
            2.01261935e-04f,
        };
    }
}
