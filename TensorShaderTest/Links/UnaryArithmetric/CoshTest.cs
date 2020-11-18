using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class CoshTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (idx + 0.5f) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = Cosh(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            4.17149087e-02f,
            1.21084251e-01f,
            1.96927071e-01f,
            2.71514218e-01f,
            3.47183188e-01f,
            4.26408069e-01f,
            5.11875095e-01f,
            6.06565963e-01f,
            7.13851346e-01f,
            8.37597311e-01f,
            9.82287751e-01f,
            1.15316640e+00f,
            1.35640257e+00f,
            1.59928547e+00f,
            1.89045265e+00f,
            2.24015946e+00f,
            2.66059696e+00f,
            3.16626788e+00f,
            3.77443116e+00f,
            4.50562810e+00f,
            5.38430510e+00f,
            6.43955100e+00f,
            7.70596999e+00f,
            9.22471515e+00f,
        };
    }
}
