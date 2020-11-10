using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Trivector {
    [TestClass]
    public class TrivectorCastTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length / 3]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length / 3]).Select((_, idx) => (float)idx * 3 - length).Reverse().ToArray();
            float[] zval = (new float[length / 3]).Select((_, idx) => (float)idx * 4 - length).ToArray();

            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).ToArray();

            ParameterField x = (Shape.Vector(length / 3), xval);
            ParameterField y = (Shape.Vector(length / 3), yval);
            ParameterField z = (Shape.Vector(length / 3), zval);
            VariableField t_actual = (Shape.Vector(length), tval);

            Field t_expect = TrivectorCast(x, y, z);
            StoreField err = t_expect - t_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");

            float[] gy_actual = y.GradState;

            AssertError.Tolerance(gy_expect, gy_actual, 1e-7f, 1e-5f, $"not equal gy");

            float[] gz_actual = z.GradState;

            AssertError.Tolerance(gz_expect, gz_actual, 1e-7f, 1e-5f, $"not equal gz");
        }

        float[] gx_expect = {
            -2.4000e+01f, -2.3500e+01f, -2.3000e+01f, -2.2500e+01f, -2.2000e+01f, -2.1500e+01f, -2.1000e+01f, -2.0500e+01f,
        };

        float[] gy_expect = {
            -3.5000e+00f, -8.0000e+00f, -1.2500e+01f, -1.7000e+01f, -2.1500e+01f, -2.6000e+01f, -3.0500e+01f, -3.5000e+01f,
        };

        float[] gz_expect = {
            -2.5000e+01f, -2.2500e+01f, -2.0000e+01f, -1.7500e+01f, -1.5000e+01f, -1.2500e+01f, -1.0000e+01f, -7.5000e+00f,
        };
    }
}
