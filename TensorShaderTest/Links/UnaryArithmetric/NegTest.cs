using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class NegTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => ((float)idx - 12) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = -x;
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -1.00000000e+00f,
            -8.75000000e-01f,
            -7.50000000e-01f,
            -6.25000000e-01f,
            -5.00000000e-01f,
            -3.75000000e-01f,
            -2.50000000e-01f,
            -1.25000000e-01f,
            -0.00000000e+00f,
            1.25000000e-01f,
            2.50000000e-01f,
            3.75000000e-01f,
            5.00000000e-01f,
            6.25000000e-01f,
            7.50000000e-01f,
            8.75000000e-01f,
            1.00000000e+00f,
            1.12500000e+00f,
            1.25000000e+00f,
            1.37500000e+00f,
            1.50000000e+00f,
            1.62500000e+00f,
            1.75000000e+00f,
            1.87500000e+00f,
        };
    }
}
