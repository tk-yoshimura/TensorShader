using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class SqrtTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (idx + 0.5f) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = Sqrt(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            5.00000000e-01f,
            4.41074435e-01f,
            4.08712907e-01f,
            3.84272488e-01f,
            3.63917237e-01f,
            3.46135636e-01f,
            3.30158445e-01f,
            3.15533803e-01f,
            3.01970491e-01f,
            2.89268192e-01f,
            2.77282298e-01f,
            2.65904454e-01f,
            2.55051026e-01f,
            2.44655885e-01f,
            2.34665685e-01f,
            2.25036659e-01f,
            2.15732378e-01f,
            2.06722137e-01f,
            1.97979775e-01f,
            1.89482786e-01f,
            1.81211643e-01f,
            1.73149277e-01f,
            1.65280659e-01f,
            1.57592483e-01f,
        };
    }
}
