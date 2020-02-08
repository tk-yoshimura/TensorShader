using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection1D {
    [TestClass]
    public class AveragePoolingTest {
        [TestMethod]
        public void ReferenceTest() {
            int stride = 3;
            int channels = 2, inwidth = 10, batch = 2;
            int outwidth = inwidth / stride;

            float[] xval = (new float[channels * inwidth * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[channels * outwidth * batch]).Select((_, idx) => idx * 2e-3f).ToArray();

            ParameterField x = new Tensor(Shape.Map1D(channels, inwidth, batch), xval);
            VariableField y_actual = new Tensor(Shape.Map1D(channels, outwidth, batch), yval);

            Field y_expect = AveragePooling1D(x, stride);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            6.66666667e-04f, 3.33333333e-04f, 6.66666667e-04f, 3.33333333e-04f, 6.66666667e-04f,
            3.33333333e-04f, 1.33333333e-03f, 1.00000000e-03f, 1.33333333e-03f, 1.00000000e-03f,
            1.33333333e-03f, 1.00000000e-03f, 2.00000000e-03f, 1.66666667e-03f, 2.00000000e-03f,
            1.66666667e-03f, 2.00000000e-03f, 1.66666667e-03f, 0.00000000e+00f, 0.00000000e+00f,
            3.33333333e-03f, 3.00000000e-03f, 3.33333333e-03f, 3.00000000e-03f, 3.33333333e-03f,
            3.00000000e-03f, 4.00000000e-03f, 3.66666667e-03f, 4.00000000e-03f, 3.66666667e-03f,
            4.00000000e-03f, 3.66666667e-03f, 4.66666667e-03f, 4.33333333e-03f, 4.66666667e-03f,
            4.33333333e-03f, 4.66666667e-03f, 4.33333333e-03f, 0.00000000e+00f, 0.00000000e+00f,
        };
    }
}
