using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Links.Utility {
    [TestClass]
    public class UnchainTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] x1val = (new float[length]).Select((_, idx) => (float)idx).ToArray();
            float[] x2val = (new float[length]).Select((_, idx) => (float)idx + 1).Reverse().ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx * 2).ToArray();

            ParameterField x1 = new Tensor(Shape.Vector(length), x1val);
            ParameterField x2 = new Tensor(Shape.Vector(length), x2val);
            VariableField y_actual = new Tensor(Shape.Vector(length), yval);

            Field y_expect = x1 + Field.Unchain(x2);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx1_actual = x1.GradState;
            Assert.IsTrue(x2.GradTensor == null);

            AssertError.Tolerance(gx1_expect, gx1_actual, 1e-7f, 1e-5f, $"not equal gx1");
        }

        float[] gx1_expect = {
            2.40000000e+01f,
            2.20000000e+01f,
            2.00000000e+01f,
            1.80000000e+01f,
            1.60000000e+01f,
            1.40000000e+01f,
            1.20000000e+01f,
            1.00000000e+01f,
            8.00000000e+00f,
            6.00000000e+00f,
            4.00000000e+00f,
            2.00000000e+00f,
            0.00000000e+00f,
            -2.00000000e+00f,
            -4.00000000e+00f,
            -6.00000000e+00f,
            -8.00000000e+00f,
            -1.00000000e+01f,
            -1.20000000e+01f,
            -1.40000000e+01f,
            -1.60000000e+01f,
            -1.80000000e+01f,
            -2.00000000e+01f,
            -2.20000000e+01f,
        };
    }
}
