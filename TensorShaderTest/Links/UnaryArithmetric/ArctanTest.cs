using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class ArctanTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (idx - 11.5f) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = Arctan(x);
            Field err = y_expect - y_actual;

            StoreField n = err;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] div = n.State;

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-6f, 1e-4f, $"not equal gx"); /*nonlinear tolerance*/
        }

        float[] gx_expect = {
            -3.98313006e-01f,
            -4.30723776e-01f,
            -4.62873034e-01f,
            -4.93626631e-01f,
            -5.21539583e-01f,
            -5.44885243e-01f,
            -5.61754767e-01f,
            -5.70246432e-01f,
            -5.68744651e-01f,
            -5.56252486e-01f,
            -5.32698251e-01f,
            -4.99109403e-01f,
            -4.57563041e-01f,
            -4.10891493e-01f,
            -3.62216732e-01f,
            -3.14455349e-01f,
            -2.69936216e-01f,
            -2.30210799e-01f,
            -1.96054354e-01f,
            -1.67598994e-01f,
            -1.44523658e-01f,
            -1.26241160e-01f,
            -1.12049086e-01f,
            -1.01234505e-01f,
        };
    }
}
