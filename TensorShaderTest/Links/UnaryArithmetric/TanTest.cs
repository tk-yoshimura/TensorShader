using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class TanTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => ((float)idx - 12) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = Tan(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -5.33492947e+00f,
            -3.63523266e+00f,
            -2.61891527e+00f,
            -1.97358690e+00f,
            -1.54384804e+00f,
            -1.24643230e+00f,
            -1.03395611e+00f,
            -8.78069451e-01f,
            -7.61063588e-01f,
            -6.71439897e-01f,
            -6.01446614e-01f,
            -5.45640512e-01f,
            -5.00000000e-01f,
            -4.61336209e-01f,
            -4.26853783e-01f,
            -3.93759599e-01f,
            -3.58827933e-01f,
            -3.17807945e-01f,
            -2.64490301e-01f,
            -1.89092527e-01f,
            -7.52736880e-02f,
            1.05714941e-01f,
            4.07202248e-01f,
            9.34240128e-01f,
        };
    }
}
