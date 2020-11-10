using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class LeftSubTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => ((float)idx - 12) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = x - 5;
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -6.00000000e+00f,
            -5.95833333e+00f,
            -5.91666667e+00f,
            -5.87500000e+00f,
            -5.83333333e+00f,
            -5.79166667e+00f,
            -5.75000000e+00f,
            -5.70833333e+00f,
            -5.66666667e+00f,
            -5.62500000e+00f,
            -5.58333333e+00f,
            -5.54166667e+00f,
            -5.50000000e+00f,
            -5.45833333e+00f,
            -5.41666667e+00f,
            -5.37500000e+00f,
            -5.33333333e+00f,
            -5.29166667e+00f,
            -5.25000000e+00f,
            -5.20833333e+00f,
            -5.16666667e+00f,
            -5.12500000e+00f,
            -5.08333333e+00f,
            -5.04166667e+00f,
        };
    }
}
