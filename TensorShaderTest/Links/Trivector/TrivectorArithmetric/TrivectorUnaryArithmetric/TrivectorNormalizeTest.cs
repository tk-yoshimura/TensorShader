using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.TrivectorArithmetric {
    [TestClass]
    public class TrivectorNormalizeTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] vval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).Reverse().ToArray();

            ParameterField v = (Shape.Vector(length), vval);
            VariableField t = (Shape.Vector(length), tval);

            Field u = TrivectorNormalize(v);
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

            ParameterField v = (Shape.Vector(length), vval);
            VariableField t = (Shape.Vector(length), tval);

            Field x = TrivectorX(v), y = TrivectorY(v), z = TrivectorZ(v);
            Field s = Sqrt(x * x + y * y + z * z);
            Field u = TrivectorCast(x / s, y / s, z / s);

            Field err = u - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gv_actual = v.GradState;

            AssertError.Tolerance(gv_expect, gv_actual, 1e-7f, 1e-5f, $"not equal gv");
        }

        float[] gv_expect = {
            1.222517518e-02f, -7.887209792e-04f, -1.380261714e-02f, 2.238993227e-02f, -2.035448388e-03f, -2.646082904e-02f,
            5.291026764e-02f, -8.140041176e-03f, -6.919034999e-02f, 2.099909758e-01f, -1.049954879e-01f, -4.199819516e-01f,
            -1.229837388e+00f, -4.919349550e-01f, 2.459674775e-01f, -1.088944443e-01f, -1.555634919e-02f, 7.778174593e-02f,
            -3.477612012e-02f, -3.024010445e-03f, 2.872809923e-02f, -1.676765222e-02f, -1.047978264e-03f, 1.467169570e-02f,
        };
    }
}
