using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class CbrtTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (idx - 11.5f) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = Cbrt(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -3.38095882e-01f,
            -3.63687299e-01f,
            -3.92787016e-01f,
            -4.26374020e-01f,
            -4.65867963e-01f,
            -5.13424545e-01f,
            -5.72516666e-01f,
            -6.49199127e-01f,
            -7.55270180e-01f,
            -9.17977774e-01f,
            -1.22222222e+00f,
            -2.23266205e+00f,
            -4.25222835e-01f,
            -5.55555556e-02f,
            8.99392950e-03f,
            2.89343408e-02f,
            3.49076436e-02f,
            3.51454784e-02f,
            3.26878579e-02f,
            2.88741097e-02f,
            2.43639919e-02f,
            1.95078363e-02f,
            1.45014484e-02f,
            9.45800934e-03f,
        };
    }
}
