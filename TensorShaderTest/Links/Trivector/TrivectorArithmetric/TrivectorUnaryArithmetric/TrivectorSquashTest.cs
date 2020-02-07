using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.TrivectorArithmetric {
    [TestClass]
    public class TrivectorSquashTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] vval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).Reverse().ToArray();

            Tensor vtensor = new Tensor(Shape.Vector(length), vval);
            Tensor ttensor = new Tensor(Shape.Vector(length), tval);

            ParameterField v = vtensor;
            VariableField t = ttensor;

            Field u = TrivectorSquash(v);
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
            Field s = Sqrt(x * x + y * y + z * z) + 1;
            Field u = TrivectorCast(x / s, y / s, z / s);

            Field err = u - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gv_actual = v.GradState;

            AssertError.Tolerance(gv_expect, gv_actual, 1e-7f, 1e-5f, $"not equal gv");
        }

        float[] gv_expect = {
            3.731359401e-03f, -8.268798143e-03f, -2.026895569e-02f, 8.107010051e-03f, -1.397115537e-02f, -3.604932080e-02f,
            2.077728172e-02f, -3.210178112e-02f, -8.498084396e-02f, 5.630698516e-02f, -1.785725023e-01f, -4.134519899e-01f,
            -1.005091987e+00f, -4.833382207e-01f, 3.841554553e-02f, -1.107038151e-01f, -2.653013547e-02f, 5.764354415e-02f,
            -3.523296143e-02f, -5.034502126e-03f, 2.516395718e-02f, -1.623129580e-02f, -9.437651812e-04f, 1.434376544e-02f,
        };
    }
}
