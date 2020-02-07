using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class SignedSqrtTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (idx - 11.5f) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = SignedSqrt(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -5.00000000e-01f,
            -5.22271770e-01f,
            -5.46829291e-01f,
            -5.74261066e-01f,
            -6.05409255e-01f,
            -6.41534629e-01f,
            -6.84637236e-01f,
            -7.38144836e-01f,
            -8.08606700e-01f,
            -9.10791918e-01f,
            -1.08925565e+00f,
            -1.62268280e+00f,
            -7.24744871e-01f,
            -2.66032346e-01f,
            -1.39009650e-01f,
            -7.86375624e-02f,
            -4.43310540e-02f,
            -2.31388367e-02f,
            -9.52466537e-03f,
            -6.93962860e-04f,
            4.92622851e-03f,
            8.29244892e-03f,
            1.00210565e-02f,
            1.05274948e-02f,
        };
    }
}
