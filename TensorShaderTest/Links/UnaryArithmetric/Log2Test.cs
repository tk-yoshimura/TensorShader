using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class Log2Test {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => idx + 0.5f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = (Shape.Vector(length), xval);
            VariableField y_actual = (Shape.Vector(length), yval);

            Field y_expect = Log2(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -2.88539008e+00f,
            5.22540137e-01f,
            7.14765808e-01f,
            6.93464315e-01f,
            6.42242414e-01f,
            5.90481515e-01f,
            5.43882651e-01f,
            5.03062746e-01f,
            4.67455174e-01f,
            4.36290345e-01f,
            4.08853009e-01f,
            3.84538272e-01f,
            3.62850059e-01f,
            3.43385020e-01f,
            3.25815031e-01f,
            3.09871936e-01f,
            2.95335191e-01f,
            2.82022122e-01f,
            2.69780336e-01f,
            2.58481793e-01f,
            2.48018185e-01f,
            2.38297305e-01f,
            2.29240166e-01f,
            2.20778723e-01f,
        };
    }
}
