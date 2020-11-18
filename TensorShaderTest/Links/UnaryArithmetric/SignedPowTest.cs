using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class SignedPowTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (idx - 11.5f) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = SignedPow(x, 1.5f);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -1.37760417e+00f,
            -1.20690090e+00f,
            -1.05132373e+00f,
            -9.10408931e-01f,
            -7.83579854e-01f,
            -6.70097939e-01f,
            -5.68980367e-01f,
            -4.78850441e-01f,
            -3.97635029e-01f,
            -3.21849115e-01f,
            -2.44408369e-01f,
            -1.42939517e-01f,
            -1.50488942e-01f,
            -2.63824630e-01f,
            -3.34276865e-01f,
            -3.78703700e-01f,
            -4.01434936e-01f,
            -4.04211734e-01f,
            -3.87873415e-01f,
            -3.52863680e-01f,
            -2.99427598e-01f,
            -2.27701267e-01f,
            -1.37757227e-01f,
            -2.96292857e-02f,
        };
    }
}
