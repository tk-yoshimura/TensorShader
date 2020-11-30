using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class NanAsZeroTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => ((float)idx - 12) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            xval[1] = xval[7] = xval[11] = float.NaN;

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = NanAsZero(x + 1);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        readonly float[] gx_expect = {
            0.00000000e+00f,
            0,
            8.33333333e-02f,
            1.25000000e-01f,
            1.66666667e-01f,
            2.08333333e-01f,
            2.50000000e-01f,
            0,
            3.33333333e-01f,
            3.75000000e-01f,
            4.16666667e-01f,
            0,
            5.00000000e-01f,
            5.41666667e-01f,
            5.83333333e-01f,
            6.25000000e-01f,
            6.66666667e-01f,
            7.08333333e-01f,
            7.50000000e-01f,
            7.91666667e-01f,
            8.33333333e-01f,
            8.75000000e-01f,
            9.16666667e-01f,
            9.58333333e-01f,
        };
    }
}
