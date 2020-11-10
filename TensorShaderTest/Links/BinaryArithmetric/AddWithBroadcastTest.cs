using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.BinaryArithmetric {
    [TestClass]
    public class AddWithBroadcastTest {
        [TestMethod]
        public void ReferenceTest() {
            int channel = 5, width = 4, height = 3, batch = 2;

            float[] x1val = (new float[channel]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] x2val = (new float[channel * width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[channel * width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

            ParameterField x1 = (Shape.Vector(channel), x1val);
            ParameterField x2 = (Shape.Map2D(channel, width, height, batch), x2val);
            VariableField y_actual = (Shape.Map2D(channel, width, height, batch), yval);

            Field y_expect = x1 + x2;
            Field err = Abs(y_expect - y_actual);

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx1_actual = x1.GradState;
            float[] gx2_actual = x2.GradState;

            AssertError.Tolerance(gx1_expect, gx1_actual, 1e-7f, 1e-5f, $"not equal gx1");

            AssertError.Tolerance(gx2_expect, gx2_actual, 1e-7f, 1e-5f, $"not equal gx2");
        }

        float[] gx1_expect = {
            0.000f * 24, 0.001f * 24, 0.002f * 24, 0.003f * 24, 0.004f * 24
        };

        float[] gx2_expect = {
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
            0.000f, 0.001f, 0.002f, 0.003f, 0.004f,
        };
    }
}
