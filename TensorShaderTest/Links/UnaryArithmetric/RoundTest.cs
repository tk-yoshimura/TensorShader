using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class RoundTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => idx - 11.3f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx + 10).ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = Round(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        readonly float[] gx_expect = {
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
            -21f,
        };
    }
}
