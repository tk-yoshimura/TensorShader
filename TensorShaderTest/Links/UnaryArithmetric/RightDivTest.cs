using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class RightDivTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (idx - 11.5f) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = 5 / x;
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            2.84047012e+01f,
            3.75898931e+01f,
            5.10511736e+01f,
            7.15896601e+01f,
            1.04533333e+02f,
            1.60855712e+02f,
            2.65604808e+02f,
            4.84444444e+02f,
            1.02717201e+03f,
            2.80800000e+03f,
            1.29333333e+04f,
            3.46920000e+05f,
            -3.44160000e+05f,
            -1.26266667e+04f,
            -2.69760000e+03f,
            -9.70845481e+02f,
            -4.50370370e+02f,
            -2.42794891e+02f,
            -1.44524351e+02f,
            -9.22666667e+01f,
            -6.20394871e+01f,
            -4.34057443e+01f,
            -3.13313897e+01f,
            -2.31873099e+01f,
        };
    }
}
