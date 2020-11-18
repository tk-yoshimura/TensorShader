using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.QuaternionArithmetric {
    [TestClass]
    public class QuaternionSquashTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).Reverse().ToArray();

            ParameterField x = xval;
            VariableField t = tval;

            Field xy = QuaternionSquash(x);
            Field err = xy - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gx_actual = x.GradState.Value;

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
            Field norm = Sqrt(Square(xr) + Square(xi) + Square(xj) + Square(xk)) + 1;

            Field xy = QuaternionCast(
                  xr / norm,
                  xi / norm,
                  xj / norm,
                  xk / norm);
            Field err = xy - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            9.711337331e-03f, -1.698325566e-03f, -1.310798846e-02f, -2.451765136e-02f, 2.436572910e-02f, -3.787894509e-03f,
            -3.194151812e-02f, -6.009514173e-02f, 8.336747012e-02f, -5.249431616e-02f, -1.883561024e-01f, -3.242178887e-01f,
            -6.483314774e-01f, -3.930048213e-01f, -1.376781653e-01f, 1.176484907e-01f, -7.346291270e-02f, -3.319310719e-02f,
            7.076698316e-03f, 4.734650382e-02f, -2.383432429e-02f, -9.303157620e-03f, 5.228009052e-03f, 1.975917572e-02f,
        };
    }
}
