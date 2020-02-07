using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.BinaryArithmetric {
    [TestClass]
    public class MulWithBroadcastTest {
        [TestMethod]
        public void ReferenceTest() {
            int channel = 5, width = 4, height = 3, batch = 2;

            float[] x1val = (new float[channel]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] x2val = (new float[channel * width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[channel * width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

            Tensor x1tensor = new Tensor(Shape.Vector(channel), x1val);
            Tensor x2tensor = new Tensor(Shape.Map2D(channel, width, height, batch), x2val);

            Tensor ytensor = new Tensor(Shape.Map2D(channel, width, height, batch), yval);

            ParameterField x1 = x1tensor;
            ParameterField x2 = x2tensor;
            VariableField y_actual = ytensor;

            Field y_expect = x1 * x2;
            Field err = Abs(y_expect - y_actual);

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx1_actual = x1.GradTensor.State;
            float[] gx2_actual = x2.GradTensor.State;

            AssertError.Tolerance(gx1_expect, gx1_actual, 1e-7f, 1e-5f, $"not equal gx1");

            AssertError.Tolerance(gx2_expect, gx2_actual, 1e-7f, 1e-5f, $"not equal gx2");
        }

        float[] gx1_expect = {
            -1.08100000e-01f,
            -1.10773116e-01f,
            -1.13488568e-01f,
            -1.16246212e-01f,
            -1.19045904e-01f
        };

        float[] gx2_expect = {
             0.00000000e+00f,
            -9.99000000e-07f,
            -3.99200000e-06f,
            -8.97300000e-06f,
            -1.59360000e-05f,
            -0.00000000e+00f,
            -5.99400000e-06f,
            -1.39720000e-05f,
            -2.39280000e-05f,
            -3.58560000e-05f,
            -0.00000000e+00f,
            -1.09890000e-05f,
            -2.39520000e-05f,
            -3.88830000e-05f,
            -5.57760000e-05f,
            -0.00000000e+00f,
            -1.59840000e-05f,
            -3.39320000e-05f,
            -5.38380000e-05f,
            -7.56960000e-05f,
            -0.00000000e+00f,
            -2.09790000e-05f,
            -4.39120000e-05f,
            -6.87930000e-05f,
            -9.56160000e-05f,
            -0.00000000e+00f,
            -2.59740000e-05f,
            -5.38920000e-05f,
            -8.37480000e-05f,
            -1.15536000e-04f,
            -0.00000000e+00f,
            -3.09690000e-05f,
            -6.38720000e-05f,
            -9.87030000e-05f,
            -1.35456000e-04f,
            -0.00000000e+00f,
            -3.59640000e-05f,
            -7.38520000e-05f,
            -1.13658000e-04f,
            -1.55376000e-04f,
            -0.00000000e+00f,
            -4.09590000e-05f,
            -8.38320000e-05f,
            -1.28613000e-04f,
            -1.75296000e-04f,
            -0.00000000e+00f,
            -4.59540000e-05f,
            -9.38120000e-05f,
            -1.43568000e-04f,
            -1.95216000e-04f,
            -0.00000000e+00f,
            -5.09490000e-05f,
            -1.03792000e-04f,
            -1.58523000e-04f,
            -2.15136000e-04f,
            -0.00000000e+00f,
            -5.59440000e-05f,
            -1.13772000e-04f,
            -1.73478000e-04f,
            -2.35056000e-04f,
            -0.00000000e+00f,
            -6.09390000e-05f,
            -1.23752000e-04f,
            -1.88433000e-04f,
            -2.54976000e-04f,
            -0.00000000e+00f,
            -6.59340000e-05f,
            -1.33732000e-04f,
            -2.03388000e-04f,
            -2.74896000e-04f,
            -0.00000000e+00f,
            -7.09290000e-05f,
            -1.43712000e-04f,
            -2.18343000e-04f,
            -2.94816000e-04f,
            -0.00000000e+00f,
            -7.59240000e-05f,
            -1.53692000e-04f,
            -2.33298000e-04f,
            -3.14736000e-04f,
            -0.00000000e+00f,
            -8.09190000e-05f,
            -1.63672000e-04f,
            -2.48253000e-04f,
            -3.34656000e-04f,
            -0.00000000e+00f,
            -8.59140000e-05f,
            -1.73652000e-04f,
            -2.63208000e-04f,
            -3.54576000e-04f,
            -0.00000000e+00f,
            -9.09090000e-05f,
            -1.83632000e-04f,
            -2.78163000e-04f,
            -3.74496000e-04f,
            -0.00000000e+00f,
            -9.59040000e-05f,
            -1.93612000e-04f,
            -2.93118000e-04f,
            -3.94416000e-04f,
            -0.00000000e+00f,
            -1.00899000e-04f,
            -2.03592000e-04f,
            -3.08073000e-04f,
            -4.14336000e-04f,
            -0.00000000e+00f,
            -1.05894000e-04f,
            -2.13572000e-04f,
            -3.23028000e-04f,
            -4.34256000e-04f,
            -0.00000000e+00f,
            -1.10889000e-04f,
            -2.23552000e-04f,
            -3.37983000e-04f,
            -4.54176000e-04f,
            -0.00000000e+00f,
            -1.15884000e-04f,
            -2.33532000e-04f,
            -3.52938000e-04f,
            -4.74096000e-04f,
        };
    }
}
