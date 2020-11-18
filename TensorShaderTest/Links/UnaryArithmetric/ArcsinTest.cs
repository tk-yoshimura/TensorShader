using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class ArcsinTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (idx - 11.5f) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = Arcsin(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-6f, 8e-4f, $"not equal gx"); /*nonlinear tolerance*/
        }

        float[] gx_expect = {
            -4.48485605e+00f,
            -2.28682106e+00f,
            -1.63165592e+00f,
            -1.29219802e+00f,
            -1.07836473e+00f,
            -9.28810443e-01f,
            -8.16983679e-01f,
            -7.29283006e-01f,
            -6.57907959e-01f,
            -5.97991798e-01f,
            -5.46279106e-01f,
            -5.00446670e-01f,
            -4.58719635e-01f,
            -4.19630103e-01f,
            -3.81841144e-01f,
            -3.43987777e-01f,
            -3.04490116e-01f,
            -2.61272952e-01f,
            -2.11257230e-01f,
            -1.49284447e-01f,
            -6.54490563e-02f,
            6.30689382e-02f,
            3.07296235e-01f,
            1.12996124e+00f,
        };
    }
}
