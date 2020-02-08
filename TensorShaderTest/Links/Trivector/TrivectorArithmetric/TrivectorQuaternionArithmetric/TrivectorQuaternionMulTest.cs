using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.TrivectorArithmetric {
    [TestClass]
    public class TrivectorQuaternionMulTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] vval = (new float[length]).Select((_, idx) => ((float)idx * 2 - length) * 0.01f).ToArray();
            float[] qval = (new float[length / 3 * 4]).Select((_, idx) => ((float)idx * 3 - length) * 0.01f).Reverse().ToArray();
            float[] tval = (new float[length]).Select((_, idx) => ((float)idx / 2) * 0.01f).ToArray();

            ParameterField v = new Tensor(Shape.Vector(length), vval);
            ParameterField q = new Tensor(Shape.Vector(length / 3 * 4), qval);
            VariableField t = new Tensor(Shape.Vector(length), tval);

            Field vq = TrivectorQuaternionMul(v, q);
            Field err = vq - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gv_actual = v.GradState;
            float[] gq_actual = q.GradState;

            AssertError.Tolerance(gv_expect, gv_actual, 1e-7f, 1e-5f, $"not equal gv");

            AssertError.Tolerance(gq_expect, gq_actual, 1e-7f, 1e-5f, $"not equal gq");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int length = 24;

            float[] vval = (new float[length]).Select((_, idx) => ((float)idx * 2 - length) * 0.01f).ToArray();
            float[] qval = (new float[length / 3 * 4]).Select((_, idx) => ((float)idx * 3 - length) * 0.01f).Reverse().ToArray();
            float[] tval = (new float[length]).Select((_, idx) => ((float)idx / 2) * 0.01f).ToArray();

            ParameterField v = new Tensor(Shape.Vector(length), vval);
            ParameterField q = new Tensor(Shape.Vector(length / 3 * 4), qval);
            VariableField t = new Tensor(Shape.Vector(length), tval);

            Field x = TrivectorX(v), y = TrivectorY(v), z = TrivectorZ(v);
            Field r = QuaternionR(q), i = QuaternionI(q), j = QuaternionJ(q), k = QuaternionK(q);

            Field xy = TrivectorCast(
                x * (r * r + i * i - j * j - k * k) + 2 * y * (i * j - r * k) + 2 * z * (r * j + k * i),
                y * (r * r - i * i + j * j - k * k) + 2 * z * (j * k - r * i) + 2 * x * (r * k + i * j),
                z * (r * r - i * i - j * j + k * k) + 2 * x * (k * i - r * j) + 2 * y * (r * i + j * k));
            Field err = xy - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gv_actual = v.GradState;

            AssertError.Tolerance(gv_expect, gv_actual, 1e-7f, 1e-5f, $"not equal gv");

            float[] gq_actual = q.GradState;

            AssertError.Tolerance(gq_expect, gq_actual, 1e-7f, 1e-5f, $"not equal gq");
        }

        float[] gv_expect = {
            -6.75738230e-01f, -6.29584711e-01f, -5.56107192e-01f, -2.42855820e-01f, -2.25015840e-01f, -1.85629860e-01f, -7.61450832e-02f, -7.18002360e-02f,
            -5.12553888e-02f, -2.37272616e-02f, -2.42301744e-02f, -1.34470872e-02f, -7.77600000e-03f, -9.05780880e-03f, -3.53561760e-03f, -6.29474400e-04f,
            -1.60729920e-03f, 1.68876000e-04f, -9.02520000e-04f, -1.88694000e-03f, -3.73536000e-03f, -9.59875920e-03f, -1.07012880e-02f, -1.58538168e-02f,
        };

        float[] gq_expect = {
            3.40467528e-01f, 3.23528592e-01f, 3.07933656e-01f, 3.02274720e-01f, 1.08645648e-01f, 1.02279456e-01f, 9.48332640e-02f, 9.55950720e-02f,
            2.75818320e-02f, 2.63670432e-02f, 2.20802544e-02f, 2.42734656e-02f, 5.04546240e-03f, 6.26678400e-03f, 2.85610560e-03f, 4.19742720e-03f,
            -1.62674400e-03f, 1.12564800e-03f, -1.88196000e-03f, -1.86556800e-03f, -3.44664000e-03f, 8.46240000e-04f, -1.31688000e-03f, -2.18400000e-03f,
            2.30944800e-03f, 8.17089600e-03f, 7.31234400e-03f, 6.02179200e-03f, 1.41848160e-02f, 2.07657792e-02f, 2.07947424e-02f, 1.86637056e-02f,
        };
    }
}
