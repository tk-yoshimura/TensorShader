using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Links.BinaryArithmetric {
    [TestClass]
    public class SubTest {
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

            Field y_expect = x1 - x2;
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx1_actual = x1.GradState;
            float[] gx2_actual = x2.GradState;

            AssertError.Tolerance(gx1_expect, gx1_actual, 1e-7f, 1e-5f, $"not equal gx1");

            AssertError.Tolerance(gx2_expect, gx2_actual, 1e-7f, 1e-5f, $"not equal gx2");
        }

        float[] gx1_expect = {
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
            -2.40000000e+01f,
        };

        float[] gx2_expect = {
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
            2.40000000e+01f,
        };
    }
}
