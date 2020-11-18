using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class MulTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => ((float)idx - 12) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = x * 5;
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -2.50000000e+01f,
            -2.31250000e+01f,
            -2.12500000e+01f,
            -1.93750000e+01f,
            -1.75000000e+01f,
            -1.56250000e+01f,
            -1.37500000e+01f,
            -1.18750000e+01f,
            -1.00000000e+01f,
            -8.12500000e+00f,
            -6.25000000e+00f,
            -4.37500000e+00f,
            -2.50000000e+00f,
            -6.25000000e-01f,
            1.25000000e+00f,
            3.12500000e+00f,
            5.00000000e+00f,
            6.87500000e+00f,
            8.75000000e+00f,
            1.06250000e+01f,
            1.25000000e+01f,
            1.43750000e+01f,
            1.62500000e+01f,
            1.81250000e+01f,
        };
    }
}
