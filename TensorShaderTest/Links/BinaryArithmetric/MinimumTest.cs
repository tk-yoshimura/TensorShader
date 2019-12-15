using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.BinaryArithmetric {
    [TestClass]
    public class MinimumTest {
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

            Field y_expect = Minimum(x1, x2);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx1_actual = x1.GradTensor.State;
            float[] gx2_actual = x2.GradTensor.State;

            AssertError.Tolerance(gx1_expect, gx1_actual, 1e-7f, 1e-5f, $"not equal gx1");

            AssertError.Tolerance(gx2_expect, gx2_actual, 1e-7f, 1e-5f, $"not equal gx2");
        }

        float[] gx1_expect = {
            0.00000000e+00f,
            -1.00000000e+00f,
            -2.00000000e+00f,
            -3.00000000e+00f,
            -4.00000000e+00f,
            -5.00000000e+00f,
            -6.00000000e+00f,
            -7.00000000e+00f,
            -8.00000000e+00f,
            -9.00000000e+00f,
            -1.00000000e+01f,
            -1.10000000e+01f,
            -1.20000000e+01f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
        };

        float[] gx2_expect = {
            0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -1.20000000e+01f,
            -1.50000000e+01f,
            -1.80000000e+01f,
            -2.10000000e+01f,
            -2.40000000e+01f,
            -2.70000000e+01f,
            -3.00000000e+01f,
            -3.30000000e+01f,
            -3.60000000e+01f,
            -3.90000000e+01f,
            -4.20000000e+01f,
            -4.50000000e+01f,
        };
    }
}
