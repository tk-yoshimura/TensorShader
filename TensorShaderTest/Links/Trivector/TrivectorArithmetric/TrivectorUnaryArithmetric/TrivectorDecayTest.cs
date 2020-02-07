using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.TrivectorArithmetric {
    [TestClass]
    public class TrivectorDecayTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] vval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).Reverse().ToArray();

            Tensor vtensor = new Tensor(Shape.Vector(length), vval);
            Tensor ttensor = new Tensor(Shape.Vector(length), tval);

            ParameterField v = vtensor;
            VariableField t = ttensor;

            Field u = TrivectorDecay(v);
            Field err = u - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gv_actual = v.GradState;

            AssertError.Tolerance(gv_expect, gv_actual, 1e-7f, 1e-5f, $"not equal gv");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int length = 24;

            float[] vval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).Reverse().ToArray();

            Tensor vtensor = new Tensor(Shape.Vector(length), vval);
            Tensor ttensor = new Tensor(Shape.Vector(length), tval);

            ParameterField v = vtensor;
            VariableField t = ttensor;

            Field x = TrivectorX(v), y = TrivectorY(v), z = TrivectorZ(v);
            Field norm = x * x + y * y + z * z;
            Field s = norm / (norm + 1);
            Field u = TrivectorCast(x * s, y * s, z * s);

            Field err = u - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gv_actual = v.GradState;

            AssertError.Tolerance(gv_expect, gv_actual, 1e-7f, 1e-5f, $"not equal gv");
        }

        float[] gv_expect = {
            -3.550846585e+01f, -3.300744665e+01f, -3.050642745e+01f, -2.801435089e+01f, -2.551196984e+01f, -2.300958880e+01f,
            -2.053294454e+01f, -1.802448722e+01f, -1.551602991e+01f, -1.316719314e+01f, -1.057929835e+01f, -7.991403563e+00f,
            -5.238095238e+00f, -3.029046539e+00f, -8.199978404e-01f, 1.995100609e+00f, 4.484346417e+00f, 6.973592226e+00f,
            9.498564964e+00f, 1.199679034e+01f, 1.449501571e+01f, 1.700010059e+01f, 1.949960630e+01f, 2.199911201e+01f,
        };
    }
}
