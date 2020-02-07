using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class EluTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - 12).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => idx / 2f).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = Elu(x, 0.5f);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -1.53604365e-06f,
            -2.26994496e-05f,
            -2.51568837e-04f,
            -2.47721612e-03f,
            -2.28106830e-02f,
            -1.98424015e-01f,
            -3.00000000e+00f,
            -1.50000000e+00f,
            0.00000000e+00f,
            1.50000000e+00f,
            3.00000000e+00f,
            4.50000000e+00f,
            6.00000000e+00f,
            7.50000000e+00f,
            9.00000000e+00f,
            1.05000000e+01f,
            1.20000000e+01f,
            1.35000000e+01f,
            1.50000000e+01f,
            1.65000000e+01f,
            1.80000000e+01f,
            1.95000000e+01f,
            2.10000000e+01f,
            2.25000000e+01f,
        };
    }
}
