using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class StepTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx - 12).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            VariableField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = Step(x);
            Field err = y_expect - y_actual;

            StoreField errnode = err.Value.Save();

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] err_actual = errnode.State;

            AssertError.Tolerance(err_expect, err_actual, 1e-7f, 1e-5f, $"not equal err");
        }

        float[] err_expect = {
            -0.00000000e+00f,
            -0.04166667e+00f,
            -0.08333333e+00f,
            -0.12500000e+00f,
            -0.16666667e+00f,
            -0.20833333e+00f,
            -0.25000000e+00f,
            -0.29166667e+00f,
            -0.33333333e+00f,
            -0.37500000e+00f,
            -0.41666667e+00f,
            -0.45833333e+00f,
            5.00000000e-01f,
            4.58333333e-01f,
            4.16666667e-01f,
            3.75000000e-01f,
            3.33333333e-01f,
            2.91666667e-01f,
            2.50000000e-01f,
            2.08333333e-01f,
            1.66666667e-01f,
            1.25000000e-01f,
            8.33333333e-02f,
            4.16666667e-02f,
        };
    }
}
