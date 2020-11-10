using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class LeftDivTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => ((float)idx - 12) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = (Shape.Vector(length), xval);
            VariableField y_actual = (Shape.Vector(length), yval);

            Field y_expect = x / 5;
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -4.00000000e-02f,
            -4.50000000e-02f,
            -5.00000000e-02f,
            -5.50000000e-02f,
            -6.00000000e-02f,
            -6.50000000e-02f,
            -7.00000000e-02f,
            -7.50000000e-02f,
            -8.00000000e-02f,
            -8.50000000e-02f,
            -9.00000000e-02f,
            -9.50000000e-02f,
            -1.00000000e-01f,
            -1.05000000e-01f,
            -1.10000000e-01f,
            -1.15000000e-01f,
            -1.20000000e-01f,
            -1.25000000e-01f,
            -1.30000000e-01f,
            -1.35000000e-01f,
            -1.40000000e-01f,
            -1.45000000e-01f,
            -1.50000000e-01f,
            -1.55000000e-01f,
        };
    }
}
