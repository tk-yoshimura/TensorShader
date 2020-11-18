using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Loss {
    [TestClass]
    public class AbsoluteErrorTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)(idx - 12) * (idx - 12) / 8 + 1).ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx * 2).ToArray();

            ParameterField x = xval;
            VariableField t = tval;

            StoreField loss = AbsoluteError(x, t);

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);

            flow.Execute();

            float[] loss_actual = loss.State.Value;

            AssertError.Tolerance(loss_expect, loss_actual, 1e-7f, 1e-5f, $"not equal loss");

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] loss_expect = {
            1.90000000e+01f,
            1.41250000e+01f,
            9.50000000e+00f,
            5.12500000e+00f,
            1.00000000e+00f,
            2.87500000e+00f,
            6.50000000e+00f,
            9.87500000e+00f,
            1.30000000e+01f,
            1.58750000e+01f,
            1.85000000e+01f,
            2.08750000e+01f,
            2.30000000e+01f,
            2.48750000e+01f,
            2.65000000e+01f,
            2.78750000e+01f,
            2.90000000e+01f,
            2.98750000e+01f,
            3.05000000e+01f,
            3.08750000e+01f,
            3.10000000e+01f,
            3.08750000e+01f,
            3.05000000e+01f,
            2.98750000e+01f,
        };

        float[] gx_expect = {
            1.90000000e+01f,
            1.41250000e+01f,
            9.50000000e+00f,
            5.12500000e+00f,
            1.00000000e+00f,
            -2.87500000e+00f,
            -6.50000000e+00f,
            -9.87500000e+00f,
            -1.30000000e+01f,
            -1.58750000e+01f,
            -1.85000000e+01f,
            -2.08750000e+01f,
            -2.30000000e+01f,
            -2.48750000e+01f,
            -2.65000000e+01f,
            -2.78750000e+01f,
            -2.90000000e+01f,
            -2.98750000e+01f,
            -3.05000000e+01f,
            -3.08750000e+01f,
            -3.10000000e+01f,
            -3.08750000e+01f,
            -3.05000000e+01f,
            -2.98750000e+01f,
        };
    }
}
