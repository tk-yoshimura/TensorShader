using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.TrivectorArithmetric {
    [TestClass]
    public class TrivectorCrossTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] vval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] uval = (new float[length]).Select((_, idx) => (float)idx * 3 - length).Reverse().ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).ToArray();

            ParameterField v = vval;
            ParameterField u = uval;
            VariableField t = tval;

            Field vu = TrivectorCross(v, u);
            Field err = vu - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gv_actual = v.GradState.Value;
            float[] gu_actual = u.GradState.Value;

            AssertError.Tolerance(gv_expect, gv_actual, 1e-7f, 1e-5f, $"not equal gv");

            AssertError.Tolerance(gu_expect, gu_actual, 1e-7f, 1e-5f, $"not equal gu");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int length = 24;

            float[] vval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] uval = (new float[length]).Select((_, idx) => (float)idx * 3 - length).Reverse().ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).ToArray();

            ParameterField v = vval;
            ParameterField u = uval;
            VariableField t = tval;

            Field vx = TrivectorX(v), vy = TrivectorY(v), vz = TrivectorZ(v);
            Field ux = TrivectorX(u), uy = TrivectorY(u), uz = TrivectorZ(u);

            Field vu = TrivectorCast(
                vy * uz - vz * uy,
                vz * ux - vx * uz,
                vx * uy - vy * ux);
            Field err = vu - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gv_actual = v.GradState.Value;

            AssertError.Tolerance(gv_expect, gv_actual, 1e-7f, 1e-5f, $"not equal gv");

            float[] gq_actual = u.GradState.Value;

            AssertError.Tolerance(gu_expect, gq_actual, 1e-7f, 1e-5f, $"not equal gq");
        }

        readonly float[] gv_expect = {
            -2.18250e+03f, 1.53000e+02f, 2.35350e+03f, -1.69650e+03f, 1.53000e+02f, 1.86750e+03f, -1.21050e+03f, 1.53000e+02f,
            1.38150e+03f, -7.24500e+02f, 1.53000e+02f, 8.95500e+02f, -2.38500e+02f, 1.53000e+02f, 4.09500e+02f, 2.47500e+02f,
            1.53000e+02f, -7.65000e+01f, 7.33500e+02f, 1.53000e+02f, -5.62500e+02f, 1.21950e+03f, 1.53000e+02f, -1.04850e+03f,
        };
        readonly float[] gu_expect = {
            -1.12800e+03f, 9.60000e+01f, 1.24800e+03f, -8.04000e+02f, 9.60000e+01f, 9.24000e+02f, -4.80000e+02f, 9.60000e+01f,
            6.00000e+02f, -1.56000e+02f, 9.60000e+01f, 2.76000e+02f, 1.68000e+02f, 9.60000e+01f, -4.80000e+01f, 4.92000e+02f,
            9.60000e+01f, -3.72000e+02f, 8.16000e+02f, 9.60000e+01f, -6.96000e+02f, 1.14000e+03f, 9.60000e+01f, -1.02000e+03f,
        };
    }
}
