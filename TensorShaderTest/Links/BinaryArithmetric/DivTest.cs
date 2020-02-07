using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Links.BinaryArithmetric {
    [TestClass]
    public class DivTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] x1val = (new float[length]).Select((_, idx) => (float)idx).ToArray();
            float[] x2val = (new float[length]).Select((_, idx) => (float)idx + 1).Reverse().ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx * 2).ToArray();

            Tensor x1tensor = new Tensor(Shape.Vector(length), x1val);
            Tensor x2tensor = new Tensor(Shape.Vector(length), x2val);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x1 = x1tensor;
            ParameterField x2 = x2tensor;
            VariableField y_actual = ytensor;

            Field y_expect = x1 / x2;
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx1_actual = x1.GradTensor.State;
            float[] gx2_actual = x2.GradTensor.State;

            AssertError.Tolerance(gx1_expect, gx1_actual, 1e-7f, 1e-5f, $"not equal gx1");

            AssertError.Tolerance(gx2_expect, gx2_actual, 1e-7f, 1e-5f, $"not equal gx2");
        }

        float[] gx1_expect = {
            0.00000000e+00f,
            -8.50661626e-02f,
            -1.77685950e-01f,
            -2.78911565e-01f,
            -3.90000000e-01f,
            -5.12465374e-01f,
            -6.48148148e-01f,
            -7.99307958e-01f,
            -9.68750000e-01f,
            -1.16000000e+00f,
            -1.37755102e+00f,
            -1.62721893e+00f,
            -1.91666667e+00f,
            -2.25619835e+00f,
            -2.66000000e+00f,
            -3.14814815e+00f,
            -3.75000000e+00f,
            -4.51020408e+00f,
            -5.50000000e+00f,
            -6.84000000e+00f,
            -8.75000000e+00f,
            -1.16666667e+01f,
            -1.65000000e+01f,
            -2.30000000e+01f,
        };

        float[] gx2_expect = {
            -0.00000000e+00f,
            3.69852881e-03f,
            1.61532682e-02f,
            3.98445092e-02f,
            7.80000000e-02f,
            1.34859309e-01f,
            2.16049383e-01f,
            3.29126806e-01f,
            4.84375000e-01f,
            6.96000000e-01f,
            9.83965015e-01f,
            1.37687756e+00f,
            1.91666667e+00f,
            2.66641623e+00f,
            3.72400000e+00f,
            5.24691358e+00f,
            7.50000000e+00f,
            1.09533528e+01f,
            1.65000000e+01f,
            2.59920000e+01f,
            4.37500000e+01f,
            8.16666667e+01f,
            1.81500000e+02f,
            5.29000000e+02f,
        };
    }
}
