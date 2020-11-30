using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.FactorArithmetric {
    [TestClass]
    public class PowTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx / 12).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = xval;
            ParameterField p = 1.5f;

            VariableField y_actual = yval;

            Field y_expect = Pow(x, p);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");

            float[] gp_actual = p.GradState.Value;

            AssertError.Tolerance(gp_expect, gp_actual, 1e-7f, 1e-5f, $"not equal gp");
        }

        readonly float[] gx_expect = {
            0.00000000e+00f,
            -7.62552925e-03f,
            -9.36436964e-03f,
            0.00000000e+00f,
            2.23290994e-02f,
            5.86987841e-02f,
            1.09834957e-01f,
            1.76270522e-01f,
            2.58418376e-01f,
            3.56610710e-01f,
            4.71122336e-01f,
            6.02185530e-01f,
            7.50000000e-01f,
            9.14739854e-01f,
            1.09655865e+00f,
            1.29559314e+00f,
            1.51196613e+00f,
            1.74578872e+00f,
            1.99716202e+00f,
            2.26617860e+00f,
            2.55292361e+00f,
            2.85747570e+00f,
            3.17990787e+00f,
            3.52028803e+00f,
        };
        readonly float[] gp_expect = {
            1.16352925e+01f
        };
    }
}
