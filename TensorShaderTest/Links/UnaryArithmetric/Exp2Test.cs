using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class Exp2Test {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => ((float)idx - 12) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = new Tensor(Shape.Vector(length), xval);
            VariableField y_actual = new Tensor(Shape.Vector(length), yval);

            Field y_expect = Exp2(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            1.73286795e-01f,
            1.79208604e-01f,
            1.85909706e-01f,
            1.93546063e-01f,
            2.02299747e-01f,
            2.12382751e-01f,
            2.24041322e-01f,
            2.37560884e-01f,
            2.53271601e-01f,
            2.71554704e-01f,
            2.92849642e-01f,
            3.17662184e-01f,
            3.46573590e-01f,
            3.80250982e-01f,
            4.19459071e-01f,
            4.65073419e-01f,
            5.18095415e-01f,
            5.79669195e-01f,
            6.51100754e-01f,
            7.33879508e-01f,
            8.29702644e-01f,
            9.40502573e-01f,
            1.06847791e+00f,
            1.21612842e+00f,
        };
    }
}
