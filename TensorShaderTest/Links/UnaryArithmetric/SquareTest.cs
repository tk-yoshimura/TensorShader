using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class SquareTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => ((float)idx - 12) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = (Shape.Vector(length), xval);
            VariableField y_actual = (Shape.Vector(length), yval);

            Field y_expect = Square(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -2.00000000e+00f,
            -1.46412037e+00f,
            -1.01851852e+00f,
            -6.56250000e-01f,
            -3.70370370e-01f,
            -1.53935185e-01f,
            -0.00000000e+00f,
            9.83796296e-02f,
            1.48148148e-01f,
            1.56250000e-01f,
            1.29629630e-01f,
            7.52314815e-02f,
            -0.00000000e+00f,
            -8.91203704e-02f,
            -1.85185185e-01f,
            -2.81250000e-01f,
            -3.70370370e-01f,
            -4.45601852e-01f,
            -5.00000000e-01f,
            -5.26620370e-01f,
            -5.18518519e-01f,
            -4.68750000e-01f,
            -3.70370370e-01f,
            -2.16435185e-01f,
        };
    }
}
