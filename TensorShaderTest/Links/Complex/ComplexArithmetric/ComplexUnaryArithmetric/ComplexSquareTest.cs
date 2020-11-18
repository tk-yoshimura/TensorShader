using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ComplexArithmetric {
    [TestClass]
    public class ComplexSquareTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 2).Reverse().ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = ComplexSquare(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 2).Reverse().ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field x_real = ComplexReal(x), x_imag = ComplexImag(x);

            Field y_expect = ComplexCast(x_real * x_real - x_imag * x_imag, 2 * x_imag * x_real);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -4.984400000e+04f,  -4.661800000e+04f,  -2.818000000e+04f,  -2.604200000e+04f,  -1.390800000e+04f,  -1.263400000e+04f,
            -5.492000000e+03f,  -4.858000000e+03f,  -1.396000000e+03f,  -1.178000000e+03f,  -8.400000000e+01f,  -5.800000000e+01f,
            -2.000000000e+01f,  3.800000000e+01f,  3.320000000e+02f,  6.460000000e+02f,  2.508000000e+03f,  3.302000000e+03f,
            8.044000000e+03f,  9.542000000e+03f,  1.847600000e+04f,  2.090200000e+04f,  3.534000000e+04f,  3.891800000e+04f,
        };
    }
}
