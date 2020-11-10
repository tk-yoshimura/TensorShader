using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ComplexArithmetric {
    [TestClass]
    public class ComplexZReluTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = { 1, 2, -3, 4, 5, -6, -1, 2, -3, -4, 5, 6, -1, 2, 3, -4, 5, -6, 1, -2, 3, -4, 5, 6 };
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 2).ToArray();

            ParameterField x = (Shape.Vector(length), xval);
            VariableField y_actual = (Shape.Vector(length), yval);

            Field y_expect = ComplexZRelu(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int length = 24;

            float[] xval = { 1, 2, -3, 4, 5, -6, -1, 2, -3, -4, 5, 6, -1, 2, 3, -4, 5, -6, 1, -2, 3, -4, 5, 6 };
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 2).ToArray();

            ParameterField x = (Shape.Vector(length), xval);
            VariableField y_actual = (Shape.Vector(length), yval);

            Field x_real = ComplexReal(x), x_imag = ComplexImag(x);
            Field isphase1 = GreaterThanOrEqual(x_real, 0) * GreaterThanOrEqual(x_imag, 0);

            Field y_expect = ComplexCast(x_real * isphase1, x_imag * isphase1);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            1f, 1.5f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0.5f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, -6f, -5.5f
        };
    }
}
