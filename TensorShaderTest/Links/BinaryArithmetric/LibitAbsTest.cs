using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.BinaryArithmetric {
    [TestClass]
    public class LimitAbsTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] x1val = (new float[length]).Select((_, idx) => (float)idx - 12).ToArray();
            float[] x2val = (new float[length]).Select((_, idx) => (float)(idx - 12) * (idx - 12) / 8 + 1).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx * 2).ToArray();

            ParameterField x1 = x1val;
            ParameterField x2 = x2val;
            VariableField y_actual = yval;

            Field y_expect = LimitAbs(x1, x2);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx1_actual = x1.GradState.Value;
            float[] gx2_actual = x2.GradState.Value;

            AssertError.Tolerance(gx1_expect, gx1_actual, 1e-7f, 1e-5f, $"not equal gx1");

            AssertError.Tolerance(gx2_expect, gx2_actual, 1e-7f, 1e-5f, $"not equal gx2");
        }

        float[] gx1_expect = {
            -1.20000000e+01f,
            -1.30000000e+01f,
            -1.40000000e+01f,
            -1.50000000e+01f,
            -1.60000000e+01f,
            -1.70000000e+01f,
            0.00000000e+00f,
            0.00000000e+00f,
            0.00000000e+00f,
            0.00000000e+00f,
            0.00000000e+00f,
            -2.30000000e+01f,
            -2.40000000e+01f,
            -2.50000000e+01f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -3.10000000e+01f,
            -3.20000000e+01f,
            -3.30000000e+01f,
            -3.40000000e+01f,
            -3.50000000e+01f,
        };

        float[] gx2_expect = {
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            1.75000000e+01f,
            1.81250000e+01f,
            1.90000000e+01f,
            2.01250000e+01f,
            2.15000000e+01f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -2.65000000e+01f,
            -2.78750000e+01f,
            -2.90000000e+01f,
            -2.98750000e+01f,
            -3.05000000e+01f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
        };
    }
}
