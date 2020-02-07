using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class LogCoshTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (idx + 0.5f) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = LogCosh(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            3.61376102e-05f,
            -4.21238672e-03f,
            -1.26892656e-02f,
            -2.35600556e-02f,
            -3.50985417e-02f,
            -4.57802434e-02f,
            -5.43459681e-02f,
            -5.98328612e-02f,
            -6.15755850e-02f,
            -5.91838432e-02f,
            -5.25041056e-02f,
            -4.15733139e-02f,
            -2.65710933e-02f,
            -7.77518871e-03f,
            1.44770169e-02f,
            3.98196301e-02f,
            6.78819220e-02f,
            9.83052004e-02f,
            1.30753916e-01f,
            1.64922167e-01f,
            2.00536731e-01f,
            2.37357596e-01f,
            2.75176817e-01f,
            3.13816319e-01f,
        };
    }
}
