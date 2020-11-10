using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class LogTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => idx + 0.5f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = Log(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -1.38629436e+00f,
            2.42532294e-01f,
            3.33182959e-01f,
            3.22217991e-01f,
            2.97202384e-01f,
            2.72075411e-01f,
            2.49508027e-01f,
            2.29764847e-01f,
            2.12556804e-01f,
            1.97504400e-01f,
            1.84257961e-01f,
            1.72522931e-01f,
            1.62058292e-01f,
            1.52668372e-01f,
            1.44194160e-01f,
            1.36505808e-01f,
            1.29496589e-01f,
            1.23078146e-01f,
            1.17176796e-01f,
            1.11730656e-01f,
            1.06687393e-01f,
            1.02002462e-01f,
            9.76377174e-02f,
            9.35603016e-02f,
        };
    }
}
