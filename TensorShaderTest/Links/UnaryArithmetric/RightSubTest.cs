using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class RightSubTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => ((float)idx - 12) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = 5 - x;
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -6.00000000e+00f,
            -5.87500000e+00f,
            -5.75000000e+00f,
            -5.62500000e+00f,
            -5.50000000e+00f,
            -5.37500000e+00f,
            -5.25000000e+00f,
            -5.12500000e+00f,
            -5.00000000e+00f,
            -4.87500000e+00f,
            -4.75000000e+00f,
            -4.62500000e+00f,
            -4.50000000e+00f,
            -4.37500000e+00f,
            -4.25000000e+00f,
            -4.12500000e+00f,
            -4.00000000e+00f,
            -3.87500000e+00f,
            -3.75000000e+00f,
            -3.62500000e+00f,
            -3.50000000e+00f,
            -3.37500000e+00f,
            -3.25000000e+00f,
            -3.12500000e+00f,
        };
    }
}
