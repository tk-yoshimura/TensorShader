using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ComplexArithmetric {
    [TestClass]
    public class ComplexRReluTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 2).ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = ComplexRRelu(x);
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
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 2).ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field x_real = ComplexReal(x), x_imag = ComplexImag(x);

            Field y_expect = ComplexCast(Relu(x_real), x_imag);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        readonly float[] gx_expect = {
            0.000000e+00f, -2.250000e+01f, 0.000000e+00f, -1.950000e+01f, 0.000000e+00f, -1.650000e+01f,
            0.000000e+00f, -1.350000e+01f, 0.000000e+00f, -1.050000e+01f, 0.000000e+00f, -7.500000e+00f,
            -6.000000e+00f, -4.500000e+00f, -3.000000e+00f, -1.500000e+00f, 0.000000e+00f, 1.500000e+00f,
            3.000000e+00f, 4.500000e+00f, 6.000000e+00f, 7.500000e+00f, 9.000000e+00f, 1.050000e+01f,
        };
    }
}
