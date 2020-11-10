using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Links.BinaryArithmetric {
    [TestClass]
    public class MulTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] x1val = (new float[length]).Select((_, idx) => (float)idx).ToArray();
            float[] x2val = (new float[length]).Select((_, idx) => (float)idx + 1).Reverse().ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx * 2).ToArray();

            ParameterField x1 = x1val;
            ParameterField x2 = x2val;
            VariableField y_actual = yval;

            Field y_expect = x1 * x2;
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx1_actual = x1.GradState;
            float[] gx2_actual = x2.GradState;

            AssertError.Tolerance(gx1_expect, gx1_actual, 1e-7f, 1e-5f, $"not equal gx1");

            AssertError.Tolerance(gx2_expect, gx2_actual, 1e-7f, 1e-5f, $"not equal gx2");
        }

        float[] gx1_expect = {
            0.00000000e+00f,
            4.83000000e+02f,
            8.80000000e+02f,
            1.19700000e+03f,
            1.44000000e+03f,
            1.61500000e+03f,
            1.72800000e+03f,
            1.78500000e+03f,
            1.79200000e+03f,
            1.75500000e+03f,
            1.68000000e+03f,
            1.57300000e+03f,
            1.44000000e+03f,
            1.28700000e+03f,
            1.12000000e+03f,
            9.45000000e+02f,
            7.68000000e+02f,
            5.95000000e+02f,
            4.32000000e+02f,
            2.85000000e+02f,
            1.60000000e+02f,
            6.30000000e+01f,
            0.00000000e+00f,
            -2.30000000e+01f,
        };

        float[] gx2_expect = {
            0.00000000e+00f,
            2.10000000e+01f,
            8.00000000e+01f,
            1.71000000e+02f,
            2.88000000e+02f,
            4.25000000e+02f,
            5.76000000e+02f,
            7.35000000e+02f,
            8.96000000e+02f,
            1.05300000e+03f,
            1.20000000e+03f,
            1.33100000e+03f,
            1.44000000e+03f,
            1.52100000e+03f,
            1.56800000e+03f,
            1.57500000e+03f,
            1.53600000e+03f,
            1.44500000e+03f,
            1.29600000e+03f,
            1.08300000e+03f,
            8.00000000e+02f,
            4.41000000e+02f,
            0.00000000e+00f,
            -5.29000000e+02f,
        };
    }
}
