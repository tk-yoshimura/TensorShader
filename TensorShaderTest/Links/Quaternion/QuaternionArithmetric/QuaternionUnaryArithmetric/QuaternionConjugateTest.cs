using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.QuaternionArithmetric {
    [TestClass]
    public class QuaternionConjugateTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).Reverse().ToArray();

            ParameterField x = xval;
            VariableField t = tval;

            Field xy = QuaternionConjugate(x);
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

            ParameterField x = xval;
            VariableField t = tval;

            Field xr = QuaternionR(x), xi = QuaternionI(x), xj = QuaternionJ(x), xk = QuaternionK(x);

            Field xy = QuaternionCast(xr, -xi, -xj, -xk);
            Field err = xy - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -3.550000000e+01f, -1.100000000e+01f, -9.500000000e+00f, -8.000000000e+00f, -2.550000000e+01f, -5.000000000e+00f,
            -3.500000000e+00f, -2.000000000e+00f, -1.550000000e+01f, 1.000000000e+00f, 2.500000000e+00f, 4.000000000e+00f,
            -5.500000000e+00f, 7.000000000e+00f, 8.500000000e+00f, 1.000000000e+01f, 4.500000000e+00f, 1.300000000e+01f,
            1.450000000e+01f, 1.600000000e+01f, 1.450000000e+01f, 1.900000000e+01f, 2.050000000e+01f, 2.200000000e+01f,
        };
    }
}
