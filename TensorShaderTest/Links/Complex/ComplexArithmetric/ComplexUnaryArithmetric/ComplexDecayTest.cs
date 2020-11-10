using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ComplexArithmetric {
    [TestClass]
    public class ComplexDecayTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 2 + 1).Reverse().ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = ComplexDecay(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

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
            Field norm = Square(x_real) + Square(x_imag);
            Field s = norm / (norm + 1);

            Field y_expect = ComplexCast(x_real * s, x_imag * s);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -3.651220333e+01f,  -3.401067586e+01f,  -3.151659464e+01f,  -2.901403862e+01f,  -2.652461671e+01f,  -2.401974602e+01f,
            -2.154219239e+01f,  -1.903073855e+01f,  -1.659544735e+01f,  -1.405549640e+01f,  -1.191469604e+01f,  -9.052586114e+00f,
            -5.200000000e+00f,  -4.928000000e+00f,  -1.548540070e+00f,  8.658691403e-01f,  3.481729916e+00f,  5.967313910e+00f,
            8.492617628e+00f,  1.098821030e+01f,  1.349695627e+01f,  1.599517735e+01f,  1.849896268e+01f,  2.099812448e+01f,
        };
    }
}
