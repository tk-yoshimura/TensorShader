using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.QuaternionArithmetric {
    [TestClass]
    public class QuaternionMulDuplicateTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).Reverse().ToArray();

            ParameterField x = (Shape.Vector(length), xval);
            VariableField t = (Shape.Vector(length), tval);

            Field xy = QuaternionMul(x, x);
            Field err = xy - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).Reverse().ToArray();

            ParameterField x = (Shape.Vector(length), xval);
            VariableField t = (Shape.Vector(length), tval);

            Field xr = QuaternionR(x), xi = QuaternionI(x), xj = QuaternionJ(x), xk = QuaternionK(x);

            Field xy = QuaternionCast(
                xr * xr - xi * xi - xj * xj - xk * xk,
                xr * xi + xi * xr + xj * xk - xk * xj,
                xr * xj - xi * xk + xj * xr + xk * xi,
                xr * xk + xi * xj - xj * xi + xk * xr);
            Field err = xy - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -8.381600000e+04f, -7.847400000e+04f, -7.131600000e+04f, -6.415800000e+04f, -2.135200000e+04f, -1.946600000e+04f,
            -1.666000000e+04f, -1.385400000e+04f, -1.640000000e+03f, -1.418000000e+03f, -9.160000000e+02f, -4.140000000e+02f,
            -1.040000000e+02f, 2.460000000e+02f, 4.920000000e+02f, 7.380000000e+02f, 7.832000000e+03f, 1.010200000e+04f,
            1.214000000e+04f, 1.417800000e+04f, 4.674400000e+04f, 5.272600000e+04f, 5.860400000e+04f, 6.448200000e+04f,
        };
    }
}
