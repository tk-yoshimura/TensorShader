using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ComplexArithmetric {
    [TestClass]
    public class ComplexRealTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length / 2]).Select((_, idx) => (float)idx / 2).ToArray();

            ParameterField x = (Shape.Vector(length), xval);
            VariableField y_actual = (Shape.Vector(length / 2), yval);

            Field y_expect = ComplexReal(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -2.400000e+01f, 0.000000e+00f, -2.050000e+01f, 0.000000e+00f, -1.700000e+01f, 0.000000e+00f,
            -1.350000e+01f, 0.000000e+00f, -1.000000e+01f, 0.000000e+00f, -6.500000e+00f, 0.000000e+00f,
            -3.000000e+00f, 0.000000e+00f, 5.000000e-01f, 0.000000e+00f, 4.000000e+00f, 0.000000e+00f,
            7.500000e+00f, 0.000000e+00f, 1.100000e+01f, 0.000000e+00f, 1.450000e+01f, 0.000000e+00f,
        };
    }
}
