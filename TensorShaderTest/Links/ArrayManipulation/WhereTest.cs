using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ArrayManipulation {
    [TestClass]
    public class WhereTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 32;

            float[] cval = (new float[length]).Select((_, idx) => idx % 3 == 0 ? 1f : 0f).ToArray();
            float[] x1val = (new float[length]).Select((_, idx) => (float)idx * 2).ToArray();
            float[] x2val = (new float[length]).Select((_, idx) => (float)idx).Reverse().ToArray();

            float[] yval = (new float[length]).Select((_, idx) => (float)(idx % 13)).ToArray();

            VariableField c = (new Shape(ShapeType.Vector, length), cval);
            ParameterField x1 = (new Shape(ShapeType.Vector, length), x1val);
            ParameterField x2 = (new Shape(ShapeType.Vector, length), x2val);
            VariableField y_actual = (new Shape(ShapeType.Vector, length), yval);

            Field y_expect = Where(c, x1, x2);
            Field err = Abs(y_expect - y_actual);

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx1_actual = x1.GradState.Value;

            AssertError.Tolerance(gx1_expect, gx1_actual, 1e-7f, 1e-5f, $"not equal gx1");

            float[] gx2_actual = x2.GradState.Value;

            AssertError.Tolerance(gx2_expect, gx2_actual, 1e-7f, 1e-5f, $"not equal gx2");
        }

        float[] gx1_expect = {
            0f,  0f,  0f,  3f,  0f,  0f,  6f,  0f,
            0f,  9f,  0f,  0f,  12f,  0f,  0f,  28f,
            0f,  0f,  31f,  0f,  0f,  34f,  0f,  0f,
            37f,  0f,  0f,  53f,  0f,  0f,  56f,  0f,
        };

        float[] gx2_expect = {
            0f,  29f,  27f,  0f,  23f,  21f,  0f,  17f,
            15f,  0f,  11f,  9f,  0f,  18f,  16f,  0f,
            12f,  10f,  0f,  6f,  4f,  0f,  0f,  -2f,
            0f,  -6f,  5f,  0f,  1f,  -1f,  0f,  -5f,
        };
    }
}
