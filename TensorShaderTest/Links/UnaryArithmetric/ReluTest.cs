using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class ReluTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 2).ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = Relu(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        readonly float[] gx_expect = {
            0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -6.00000000e+00f,
            -4.50000000e+00f,
            -3.00000000e+00f,
            -1.50000000e+00f,
            0.00000000e+00f,
            1.50000000e+00f,
            3.00000000e+00f,
            4.50000000e+00f,
            6.00000000e+00f,
            7.50000000e+00f,
            9.00000000e+00f,
            1.05000000e+01f
        };
    }
}
