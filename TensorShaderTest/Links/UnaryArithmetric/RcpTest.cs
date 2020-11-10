using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class RcpTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (idx - 11.5f) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = (Shape.Vector(length), xval);
            VariableField y_actual = (Shape.Vector(length), yval);

            Field y_expect = Rcp(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            1.13618805e+00f,
            1.54713314e+00f,
            2.14841814e+00f,
            3.06289436e+00f,
            4.52266667e+00f,
            7.00227583e+00f,
            1.15762585e+01f,
            2.10370370e+01f,
            4.42215743e+01f,
            1.19232000e+02f,
            5.38666667e+02f,
            1.40880000e+04f,
            -1.35360000e+04f,
            -4.77333333e+02f,
            -9.71520000e+01f,
            -3.29562682e+01f,
            -1.42222222e+01f,
            -7.01427498e+00f,
            -3.73600364e+00f,
            -2.06933333e+00f,
            -1.15285976e+00f,
            -6.19332264e-01f,
            -2.95432459e-01f,
            -9.27097888e-02f,
        };
    }
}
