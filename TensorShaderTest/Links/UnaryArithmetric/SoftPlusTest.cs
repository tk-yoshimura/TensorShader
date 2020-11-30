using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class SoftPlusTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - 12).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => idx / 2f).ToArray();

            ParameterField x = xval;
            VariableField y_actual = yval;

            Field y_expect = SoftPlus(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        readonly float[] gx_expect = {
            3.77509975e-11f,
            -2.26968733e-05f,
            -3.35237652e-04f,
            -3.70281330e-03f,
            -3.56459715e-02f,
            -2.82877115e-01f,
            -1.15342641e+00f,
            -1.20939780e+00f,
            1.78234795e-02f,
            1.49876063e+00f,
            2.99932924e+00f,
            4.49984111e+00f,
            5.99996928e+00f,
            7.49999460e+00f,
            8.99999910e+00f,
            1.04999999e+01f,
            1.20000000e+01f,
            1.35000000e+01f,
            1.50000000e+01f,
            1.65000000e+01f,
            1.80000000e+01f,
            1.95000000e+01f,
            2.10000000e+01f,
            2.25000000e+01f,
        };
    }
}
