using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection1D {
    [TestClass]
    public class EdgePaddingTest {
        [TestMethod]
        public void ReferenceTest() {
            int pad_left = 2, pad_right = 1;
            int channels = 5, inwidth = 6, batch = 2;
            int outwidth = inwidth + pad_left + pad_right;

            float[] xval = (new float[channels * inwidth * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[channels * outwidth * batch]).Select((_, idx) => idx * 2e-3f).ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(channels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(channels, outwidth, batch), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = EdgePadding1D(x, pad_left, pad_right);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -2.0000e-02f, -2.1000e-02f, -2.2000e-02f, -2.3000e-02f, -2.4000e-02f,
            -2.5000e-02f, -2.6000e-02f, -2.7000e-02f, -2.8000e-02f, -2.9000e-02f,
            -3.0000e-02f, -3.1000e-02f, -3.2000e-02f, -3.3000e-02f, -3.4000e-02f,
            -3.5000e-02f, -3.6000e-02f, -3.7000e-02f, -3.8000e-02f, -3.9000e-02f,
            -4.0000e-02f, -4.1000e-02f, -4.2000e-02f, -4.3000e-02f, -4.4000e-02f,
            -4.5000e-02f, -4.6000e-02f, -4.7000e-02f, -4.8000e-02f, -4.9000e-02f,
            -8.0000e-02f, -8.1000e-02f, -8.2000e-02f, -8.3000e-02f, -8.4000e-02f,
            -8.5000e-02f, -8.6000e-02f, -8.7000e-02f, -8.8000e-02f, -8.9000e-02f,
            -9.0000e-02f, -9.1000e-02f, -9.2000e-02f, -9.3000e-02f, -9.4000e-02f,
            -9.5000e-02f, -9.6000e-02f, -9.7000e-02f, -9.8000e-02f, -9.9000e-02f,
            -1.0000e-01f, -1.0100e-01f, -1.0200e-01f, -1.0300e-01f, -1.0400e-01f,
            -1.0500e-01f, -1.0600e-01f, -1.0700e-01f, -1.0800e-01f, -1.0900e-01f,
        };
    }
}
