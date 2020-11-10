using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.TrinaryArithmetric {
    [TestClass]
    public class ClampTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx - 12).ToArray();
            float[] xminval = (new float[length]).Select((_, idx) => (float)(idx - 12) * (idx - 12) / 8 - 1).ToArray();
            float[] xmaxval = (new float[length]).Select((_, idx) => (float)(idx - 12) * (idx - 12) / 8 + 1).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx * 2).ToArray();

            ParameterField x = xval;
            ParameterField xmin = xminval;
            ParameterField xmax = xmaxval;
            VariableField y_actual = yval;

            Field y_expect = Clamp(x, xmin, xmax);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;
            float[] gxmin_actual = xmin.GradState;
            float[] gxmax_actual = xmax.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
            AssertError.Tolerance(gxmin_expect, gxmin_actual, 1e-7f, 1e-5f, $"not equal gxmin");
            AssertError.Tolerance(gxmax_expect, gxmax_actual, 1e-7f, 1e-5f, $"not equal gxmax");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx - 12).ToArray();
            float[] xminval = (new float[length]).Select((_, idx) => (float)(idx - 12) * (idx - 12) / 8 - 1).ToArray();
            float[] xmaxval = (new float[length]).Select((_, idx) => (float)(idx - 12) * (idx - 12) / 8 + 1).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx * 2).ToArray();

            ParameterField x = xval;
            ParameterField xmin = xminval;
            ParameterField xmax = xmaxval;
            VariableField y_actual = yval;

            Field y_expect = Minimum(Maximum(x, xmin), xmax);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;
            float[] gxmin_actual = xmin.GradState;
            float[] gxmax_actual = xmax.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
            AssertError.Tolerance(gxmin_expect, gxmin_actual, 1e-7f, 1e-5f, $"not equal gxmin");
            AssertError.Tolerance(gxmax_expect, gxmax_actual, 1e-7f, 1e-5f, $"not equal gxmax");
        }

        float[] gx_expect = {
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -24f,
            -25f,
            0,
            0,
            0,
            0,
            0,
            -31f,
            -32f,
            0,
            0,
            0,
        };

        float[] gxmin_expect = {
            17f,
            12.125f,
            7.5f,
            3.125f,
            -1f,
            -4.875f,
            -8.5f,
            -11.875f,
            -15f,
            -17.875f,
            -20.5f,
            -22.875f,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -32.875f,
            -32.5f,
            -31.875f,
        };

        float[] gxmax_expect = {
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -26.5f,
            -27.875f,
            -29f,
            -29.875f,
            -30.5f,
            0,
            0,
            0,
            0,
            0,
        };
    }
}
