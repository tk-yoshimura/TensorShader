using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.QuaternionArithmetric {
    [TestClass]
    public class QuaternionMulTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx * 3 - length).Reverse().ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).ToArray();

            ParameterField x = xval;
            ParameterField y = yval;
            VariableField t = tval;

            Field xy = QuaternionMul(x, y);
            Field err = xy - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gx_actual = x.GradState.Value;
            float[] gy_actual = y.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");

            AssertError.Tolerance(gy_expect, gy_actual, 1e-7f, 1e-5f, $"not equal gy");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx * 3 - length).Reverse().ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).ToArray();

            ParameterField x = xval;
            ParameterField y = yval;
            VariableField t = tval;

            Field xr = QuaternionR(x), xi = QuaternionI(x), xj = QuaternionJ(x), xk = QuaternionK(x);
            Field yr = QuaternionR(y), yi = QuaternionI(y), yj = QuaternionJ(y), yk = QuaternionK(y);

            Field xy = QuaternionCast(
                xr * yr - xi * yi - xj * yj - xk * yk,
                xr * yi + xi * yr + xj * yk - xk * yj,
                xr * yj - xi * yk + xj * yr + xk * yi,
                xr * yk + xi * yj - xj * yi + xk * yr);
            Field err = xy - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");

            float[] gy_actual = y.GradState.Value;

            AssertError.Tolerance(gy_expect, gy_actual, 1e-7f, 1e-5f, $"not equal gy");
        }

        readonly float[] gx_expect = {
            -1.586580e+05f, -1.453770e+05f, -1.321200e+05f, -1.189980e+05f, -5.301000e+04f, -4.616100e+04f,
            -3.952800e+04f, -3.303000e+04f, -9.378000e+03f, -6.849000e+03f, -4.536000e+03f, -2.358000e+03f,
            -1.140000e+02f, 2.070000e+02f, 5.040000e+02f, 6.660000e+02f, 2.430000e+03f, 2.655000e+03f,
            3.240000e+03f, 3.690000e+03f, 2.590200e+04f, 2.814300e+04f, 3.132000e+04f, 3.436200e+04f,
        };
        readonly float[] gy_expect = {
            8.033800e+04f, 7.492800e+04f, 6.962400e+04f, 6.424800e+04f, 2.310600e+04f, 2.088000e+04f,
            1.884000e+04f, 1.672800e+04f, 2.610000e+03f, 2.160000e+03f, 1.848000e+03f, 1.464000e+03f,
            4.180000e+02f, 3.360000e+02f, 2.160000e+02f, 2.400000e+01f, -1.902000e+03f, -3.024000e+03f,
            -4.488000e+03f, -6.024000e+03f, -2.278200e+04f, -2.635200e+04f, -3.069600e+04f, -3.511200e+04f,
        };
    }
}
