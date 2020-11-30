using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection2D {
    [TestClass]
    public class LinearZoomTest {
        [TestMethod]
        public void ReferenceTest() {
            int scale = 2;
            int channels = 2, inwidth = 4, inheight = 3, batch = 2;
            int outwidth = inwidth * scale, outheight = inheight * scale;

            float[] xval = (new float[channels * inwidth * inheight * batch]).Select((_, idx) => idx * 2e-3f).ToArray();
            float[] yval = (new float[channels * outwidth * outheight * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

            ParameterField x = (Shape.Map2D(channels, inwidth, inheight, batch), xval);
            VariableField y_actual = (Shape.Map2D(channels, outwidth, outheight, batch), yval);

            Field y_expect = LinearZoom2D(x);
            Field err = Abs(y_expect - y_actual);

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        readonly float[] gx_expect = {
            -5.666667e-03f, -4.666667e-03f, -6.333333e-03f, -5.333333e-03f, -6.333333e-03f, -5.333333e-03f, -7.000000e-03f, -6.000000e-03f,
            -2.433333e-02f, -2.333333e-02f, -2.500000e-02f, -2.400000e-02f, -2.500000e-02f, -2.400000e-02f, -2.566667e-02f, -2.466667e-02f,
            -4.300000e-02f, -4.200000e-02f, -4.366667e-02f, -4.266667e-02f, -4.366667e-02f, -4.266667e-02f, -4.433333e-02f, -4.333333e-02f,
            -5.366667e-02f, -5.266667e-02f, -5.433333e-02f, -5.333333e-02f, -5.433333e-02f, -5.333333e-02f, -5.500000e-02f, -5.400000e-02f,
            -7.233333e-02f, -7.133333e-02f, -7.300000e-02f, -7.200000e-02f, -7.300000e-02f, -7.200000e-02f, -7.366667e-02f, -7.266667e-02f,
            -9.100000e-02f, -9.000000e-02f, -9.166667e-02f, -9.066667e-02f, -9.166667e-02f, -9.066667e-02f, -9.233333e-02f, -9.133333e-02f,
        };
    }
}
