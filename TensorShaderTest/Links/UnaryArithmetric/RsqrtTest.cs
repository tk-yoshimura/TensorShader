using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class RsqrtTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (idx + 0.5f) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = (Shape.Vector(length), xval);
            VariableField y_actual = (Shape.Vector(length), yval);

            Field y_expect = Rsqrt(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -2.88000000e+02f,
            -3.15285955e+01f,
            -1.10818220e+01f,
            -5.48077098e+00f,
            -3.19266819e+00f,
            -2.04446122e+00f,
            -1.39058837e+00f,
            -9.84854085e-01f,
            -7.16968721e-01f,
            -5.31596387e-01f,
            -3.98526708e-01f,
            -3.00149827e-01f,
            -2.25648985e-01f,
            -1.68089181e-01f,
            -1.22862445e-01f,
            -8.68129808e-02f,
            -5.77227213e-02f,
            -3.39972207e-02f,
            -1.44675240e-02f,
            1.73840386e-03f,
            1.52812271e-02f,
            2.66684405e-02f,
            3.62947595e-02f,
            4.44706561e-02f,
        };
    }
}
