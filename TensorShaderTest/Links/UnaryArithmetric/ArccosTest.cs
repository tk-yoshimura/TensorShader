using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class ArccosTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (idx - 11.5f) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = Arccos(x);
            Field err = y_expect - y_actual;

            OutputNode n = err.Value.Save();

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] div = n.Tensor.State;

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-6f, 8e-4f, $"not equal gx"); /*nonlinear tolerance*/
        }

        float[] gx_expect = {
            -9.98383667e+00f,
            -5.35931140e+00f,
            -3.92991627e+00f,
            -3.16333743e+00f,
            -2.66358605e+00f,
            -2.30180444e+00f,
            -2.02177663e+00f,
            -1.79447887e+00f,
            -1.60313612e+00f,
            -1.43720218e+00f,
            -1.28957194e+00f,
            -1.15514489e+00f,
            -1.03001209e+00f,
            -9.10946619e-01f,
            -7.95037201e-01f,
            -6.79366357e-01f,
            -5.60646144e-01f,
            -4.34689990e-01f,
            -2.95479077e-01f,
            -1.33224225e-01f,
            7.03680773e-02f,
            3.56387076e-01f,
            8.49590343e-01f,
            2.34077022e+00f,
        };
    }
}
