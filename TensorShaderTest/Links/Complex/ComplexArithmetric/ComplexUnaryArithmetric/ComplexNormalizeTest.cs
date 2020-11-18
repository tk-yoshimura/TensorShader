using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ComplexArithmetric {
    [TestClass]
    public class ComplexNormalizeTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 2 + 1).Reverse().ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = ComplexNormalize(x);
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
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 2 + 1).Reverse().ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field x_real = ComplexReal(x), x_imag = ComplexImag(x);
            Field norm = Sqrt(Square(x_real) + Square(x_imag));

            Field y_expect = ComplexCast(x_real / norm, x_imag / norm);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            8.287188840e-03f,  -9.040569644e-03f,  1.201179767e-02f,  -1.334644185e-02f,  1.893929182e-02f,  -2.164490494e-02f,
            3.410818523e-02f,  -4.092982227e-02f,  7.800000000e-02f,  -1.040000000e-01f,  2.906888371e-01f,  -5.813776741e-01f,
            -3.250000000e+00f,  0.000000000e+00f,  -2.080125736e-01f,  1.386750491e-01f,  -6.189813733e-02f,  4.951850987e-02f,
            -2.903039950e-02f,  2.488319957e-02f,  -1.675227783e-02f,  1.489091363e-02f,  -1.088147167e-02f,  9.892246971e-03f,
        };
    }
}
