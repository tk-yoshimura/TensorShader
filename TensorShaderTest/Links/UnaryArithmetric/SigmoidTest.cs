using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class SigmoidTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx - 12).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = Sigmoid(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        readonly float[] gx_expect = {
            3.77506496e-11f,
            -6.95602022e-07f,
            -3.78092311e-06f,
            -1.54071944e-05f,
            -5.57605231e-05f,
            -1.88800154e-04f,
            -6.10528575e-04f,
            -1.89452208e-03f,
            -5.56988360e-03f,
            -1.47987049e-02f,
            -3.12317851e-02f,
            -3.72367100e-02f,
            0.00000000e+00f,
            3.72367100e-02f,
            3.12317851e-02f,
            1.47987049e-02f,
            5.56988360e-03f,
            1.89452208e-03f,
            6.10528575e-04f,
            1.88800154e-04f,
            5.57605231e-05f,
            1.54071944e-05f,
            3.78092311e-06f,
            6.95602022e-07f,
        };
    }
}
