using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ArrayManipulation {
    [TestClass]
    public class SumTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] x1val = (new float[length]).Select((_, idx) => (float)idx).ToArray();
            float[] x2val = (new float[length]).Select((_, idx) => (float)idx + 1).Reverse().ToArray();
            float[] x3val = (new float[length]).Select((_, idx) => (float)idx % 5).ToArray();

            float[] yval = (new float[length]).Select((_, idx) => (float)idx * 2).ToArray();

            ParameterField x1 = new Tensor(Shape.Vector(length), x1val);
            ParameterField x2 = new Tensor(Shape.Vector(length), x2val);
            ParameterField x3 = new Tensor(Shape.Vector(length), x3val);
            VariableField y_actual = new Tensor(Shape.Vector(length), yval);

            Field y_expect = Sum(x1, x2, x3);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx1_actual = x1.GradState;
            float[] gx2_actual = x2.GradState;
            float[] gx3_actual = x3.GradState;

            AssertError.Tolerance(gx_expect, gx1_actual, 1e-7f, 1e-5f, $"not equal gx1");

            AssertError.Tolerance(gx_expect, gx2_actual, 1e-7f, 1e-5f, $"not equal gx2");

            AssertError.Tolerance(gx_expect, gx3_actual, 1e-7f, 1e-5f, $"not equal gx3");
        }

        float[] gx_expect = {
            2.40000000e+01f,
            2.30000000e+01f,
            2.20000000e+01f,
            2.10000000e+01f,
            2.00000000e+01f,
            1.40000000e+01f,
            1.30000000e+01f,
            1.20000000e+01f,
            1.10000000e+01f,
            1.00000000e+01f,
            4.00000000e+00f,
            3.00000000e+00f,
            2.00000000e+00f,
            1.00000000e+00f,
            0.00000000e+00f,
            -6.00000000e+00f,
            -7.00000000e+00f,
            -8.00000000e+00f,
            -9.00000000e+00f,
            -1.00000000e+01f,
            -1.60000000e+01f,
            -1.70000000e+01f,
            -1.80000000e+01f,
            -1.90000000e+01f,
        };
    }
}
