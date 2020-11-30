using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection1D {
    [TestClass]
    public class TrimmingTest {
        [TestMethod]
        public void ReferenceTest() {
            int trim_left = 2, trim_right = 1;
            int channels = 5, inwidth = 9, batch = 2;
            int outwidth = inwidth - trim_left - trim_right;

            float[] xval = (new float[channels * inwidth * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[channels * outwidth * batch]).Select((_, idx) => idx * 2e-3f).ToArray();

            ParameterField x = (Shape.Map1D(channels, inwidth, batch), xval);
            VariableField y_actual = (Shape.Map1D(channels, outwidth, batch), yval);

            Field y_expect = Trimming1D(x, trim_left, trim_right);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        readonly float[] gx_expect = {
            0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f,
            0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f,
            1.0000e-02f, 9.0000e-03f, 8.0000e-03f, 7.0000e-03f, 6.0000e-03f,
            5.0000e-03f, 4.0000e-03f, 3.0000e-03f, 2.0000e-03f, 1.0000e-03f,
            0.0000e+00f, -1.0000e-03f, -2.0000e-03f, -3.0000e-03f, -4.0000e-03f,
            -5.0000e-03f, -6.0000e-03f, -7.0000e-03f, -8.0000e-03f, -9.0000e-03f,
            -1.0000e-02f, -1.1000e-02f, -1.2000e-02f, -1.3000e-02f, -1.4000e-02f,
            -1.5000e-02f, -1.6000e-02f, -1.7000e-02f, -1.8000e-02f, -1.9000e-02f,
            0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f,
            0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f,
            0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f,
            -5.0000e-03f, -6.0000e-03f, -7.0000e-03f, -8.0000e-03f, -9.0000e-03f,
            -1.0000e-02f, -1.1000e-02f, -1.2000e-02f, -1.3000e-02f, -1.4000e-02f,
            -1.5000e-02f, -1.6000e-02f, -1.7000e-02f, -1.8000e-02f, -1.9000e-02f,
            -2.0000e-02f, -2.1000e-02f, -2.2000e-02f, -2.3000e-02f, -2.4000e-02f,
            -2.5000e-02f, -2.6000e-02f, -2.7000e-02f, -2.8000e-02f, -2.9000e-02f,
            -3.0000e-02f, -3.1000e-02f, -3.2000e-02f, -3.3000e-02f, -3.4000e-02f,
            0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f,
        };
    }
}
