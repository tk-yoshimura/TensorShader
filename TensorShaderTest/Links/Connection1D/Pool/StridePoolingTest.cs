using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection1D {
    [TestClass]
    public class StridePoolingTest {
        [TestMethod]
        public void ReferenceTest() {
            int stride = 3;
            int channels = 2, inwidth = 10, batch = 2;
            int outwidth = inwidth / stride;

            float[] xval = (new float[channels * inwidth * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[channels * outwidth * batch]).Select((_, idx) => idx * 2e-3f).ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(channels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(channels, outwidth, batch), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = StridePooling1D(x, stride);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            0.00000000e+00f, -1.00000000e-03f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f,
            0.00000000e+00f, 2.00000000e-03f, 1.00000000e-03f, 0.00000000e+00f, 0.00000000e+00f,
            0.00000000e+00f, 0.00000000e+00f, 4.00000000e-03f, 3.00000000e-03f, 0.00000000e+00f,
            0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f,
            8.00000000e-03f, 7.00000000e-03f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f,
            0.00000000e+00f, 1.00000000e-02f, 9.00000000e-03f, 0.00000000e+00f, 0.00000000e+00f,
            0.00000000e+00f, 0.00000000e+00f, 1.20000000e-02f, 1.10000000e-02f, 0.00000000e+00f,
            0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f,
        };
    }
}
