using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection2D {
    [TestClass]
    public class NeighborZoomTest {
        [TestMethod]
        public void ReferenceTest() {
            int scale = 2;
            int channels = 2, inwidth = 4, inheight = 3, batch = 2;
            int outwidth = inwidth * scale, outheight = inheight * scale;

            float[] xval = (new float[channels * inwidth * inheight * batch]).Select((_, idx) => idx * 2e-3f).ToArray();
            float[] yval = (new float[channels * outwidth * outheight * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

            ParameterField x = (Shape.Map2D(channels, inwidth, inheight, batch), xval);
            VariableField y_actual = (Shape.Map2D(channels, outwidth, outheight, batch), yval);

            Field y_expect = NeighborZoom2D(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -9.0000e-03f, -8.0000e-03f, -9.0000e-03f, -8.0000e-03f, -9.0000e-03f, -8.0000e-03f, -9.0000e-03f, -8.0000e-03f,
            -2.5000e-02f, -2.4000e-02f, -2.5000e-02f, -2.4000e-02f, -2.5000e-02f, -2.4000e-02f, -2.5000e-02f, -2.4000e-02f,
            -4.1000e-02f, -4.0000e-02f, -4.1000e-02f, -4.0000e-02f, -4.1000e-02f, -4.0000e-02f, -4.1000e-02f, -4.0000e-02f,
            -5.7000e-02f, -5.6000e-02f, -5.7000e-02f, -5.6000e-02f, -5.7000e-02f, -5.6000e-02f, -5.7000e-02f, -5.6000e-02f,
            -7.3000e-02f, -7.2000e-02f, -7.3000e-02f, -7.2000e-02f, -7.3000e-02f, -7.2000e-02f, -7.3000e-02f, -7.2000e-02f,
            -8.9000e-02f, -8.8000e-02f, -8.9000e-02f, -8.8000e-02f, -8.9000e-02f, -8.8000e-02f, -8.9000e-02f, -8.8000e-02f,
        };
    }
}
