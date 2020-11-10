using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class SignTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx - 12).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = (Shape.Vector(length), xval);
            VariableField y_actual = (Shape.Vector(length), yval);

            Field y_expect = Sign(x);
            StoreField err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] err_actual = err.State;

            AssertError.Tolerance(err_expect, err_actual, 1e-7f, 1e-5f, $"not equal err");
        }

        float[] err_expect = {
            -1.00000000e+00f,
            -1.04166667e+00f,
            -1.08333333e+00f,
            -1.12500000e+00f,
            -1.16666667e+00f,
            -1.20833333e+00f,
            -1.25000000e+00f,
            -1.29166667e+00f,
            -1.33333333e+00f,
            -1.37500000e+00f,
            -1.41666667e+00f,
            -1.45833333e+00f,
            -5.00000000e-01f,
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
