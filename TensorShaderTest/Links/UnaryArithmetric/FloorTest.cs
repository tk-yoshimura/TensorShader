using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class FloorTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => idx - 11.5f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx + 10).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = Floor(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
            -22f,
        };
    }
}
