using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ArrayManipulation {
    [TestClass]
    public class BroadcastTest {
        [TestMethod]
        public void ReferenceTest() {
            int s0 = 3, s1 = 6, s2 = 4, s3 = 5, s4 = 1, s5 = 2, s6 = 2, s7 = 3;

            float[] xval = (new float[s0 * s2 * s3 * s5]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[s0 * s1 * s2 * s3 * s4 * s5 * s6 * s7]).Select((_, idx) => idx * 1e-3f).ToArray();

            ParameterField x = (new Shape(ShapeType.Map, s0, 1, s2, s3, 1, s5), xval);
            VariableField y_actual = (new Shape(ShapeType.Map, s0, s1, s2, s3, s4, s5, s6, s7), yval);

            Field y_expect = Broadcast(x, y_actual.Shape);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -6.5070e+01f, -6.5070e+01f, -6.5070e+01f, -6.5610e+01f, -6.5610e+01f,
            -6.5610e+01f, -6.6150e+01f, -6.6150e+01f, -6.6150e+01f, -6.6690e+01f,
            -6.6690e+01f, -6.6690e+01f, -6.7230e+01f, -6.7230e+01f, -6.7230e+01f,
            -6.7770e+01f, -6.7770e+01f, -6.7770e+01f, -6.8310e+01f, -6.8310e+01f,
            -6.8310e+01f, -6.8850e+01f, -6.8850e+01f, -6.8850e+01f, -6.9390e+01f,
            -6.9390e+01f, -6.9390e+01f, -6.9930e+01f, -6.9930e+01f, -6.9930e+01f,
            -7.0470e+01f, -7.0470e+01f, -7.0470e+01f, -7.1010e+01f, -7.1010e+01f,
            -7.1010e+01f, -7.1550e+01f, -7.1550e+01f, -7.1550e+01f, -7.2090e+01f,
            -7.2090e+01f, -7.2090e+01f, -7.2630e+01f, -7.2630e+01f, -7.2630e+01f,
            -7.3170e+01f, -7.3170e+01f, -7.3170e+01f, -7.3710e+01f, -7.3710e+01f,
            -7.3710e+01f, -7.4250e+01f, -7.4250e+01f, -7.4250e+01f, -7.4790e+01f,
            -7.4790e+01f, -7.4790e+01f, -7.5330e+01f, -7.5330e+01f, -7.5330e+01f,
            -7.5870e+01f, -7.5870e+01f, -7.5870e+01f, -7.6410e+01f, -7.6410e+01f,
            -7.6410e+01f, -7.6950e+01f, -7.6950e+01f, -7.6950e+01f, -7.7490e+01f,
            -7.7490e+01f, -7.7490e+01f, -7.8030e+01f, -7.8030e+01f, -7.8030e+01f,
            -7.8570e+01f, -7.8570e+01f, -7.8570e+01f, -7.9110e+01f, -7.9110e+01f,
            -7.9110e+01f, -7.9650e+01f, -7.9650e+01f, -7.9650e+01f, -8.0190e+01f,
            -8.0190e+01f, -8.0190e+01f, -8.0730e+01f, -8.0730e+01f, -8.0730e+01f,
            -8.1270e+01f, -8.1270e+01f, -8.1270e+01f, -8.1810e+01f, -8.1810e+01f,
            -8.1810e+01f, -8.2350e+01f, -8.2350e+01f, -8.2350e+01f, -8.2890e+01f,
            -8.2890e+01f, -8.2890e+01f, -8.3430e+01f, -8.3430e+01f, -8.3430e+01f,
            -8.3970e+01f, -8.3970e+01f, -8.3970e+01f, -8.4510e+01f, -8.4510e+01f,
            -8.4510e+01f, -8.5050e+01f, -8.5050e+01f, -8.5050e+01f, -8.5590e+01f,
            -8.5590e+01f, -8.5590e+01f, -8.6130e+01f, -8.6130e+01f, -8.6130e+01f,
        };
    }
}
