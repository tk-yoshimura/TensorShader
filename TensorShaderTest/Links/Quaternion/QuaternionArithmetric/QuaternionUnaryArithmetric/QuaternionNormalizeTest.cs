using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.QuaternionArithmetric {
    [TestClass]
    public class QuaternionNormalizeTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).Reverse().ToArray();

            ParameterField x = (Shape.Vector(length), xval);
            VariableField t = (Shape.Vector(length), tval);

            Field xy = QuaternionNormalize(x);
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
            Field norm = Sqrt(Square(xr) + Square(xi) + Square(xj) + Square(xk));

            Field xy = QuaternionCast(
                  xr / norm,
                  xi / norm,
                  xj / norm,
                  xk / norm);
            Field err = xy - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            1.693395492e-02f, 4.671435840e-03f, -7.591083240e-03f, -1.985360232e-02f, 4.073687763e-02f, 9.585147677e-03f,
            -2.156658227e-02f, -5.271831223e-02f, 1.673596703e-01f, 0.000000000e+00f, -1.673596703e-01f, -3.347193407e-01f,
            -7.349684153e-01f, -4.199819516e-01f, -1.049954879e-01f, 2.099909758e-01f, -7.388571370e-02f, -3.110977419e-02f,
            1.166616532e-02f, 5.444210483e-02f, -2.435021248e-02f, -9.425888702e-03f, 5.498435076e-03f, 2.042275885e-02f,
        };
    }
}
