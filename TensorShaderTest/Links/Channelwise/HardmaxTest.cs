using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Channelwise {
    [TestClass]
    public class HardmaxTest {
        [TestMethod]
        public void ReferenceMethod() {
            int channels = 3, indexes = 4;

            float[] xval =
                { 1, 2, 3,
                  -2, 1, 3,
                  3, 2, -1,
                  1, 2, 2 };

            float[] tval =
                { 0, 0, 1,
                  0, 1, 0,
                  0, 1, 0,
                  0, 1, 1 };

            ParameterField x = (Shape.Map0D(channels, indexes), xval);
            VariableField t = (Shape.Map0D(channels, indexes), tval);

            StoreField y = Hardmax(x);
            Field err = y - t;
            (Flow flow, _) = Flow.Optimize(err);

            flow.Execute();

            float[] y_actual = y.State.Value;

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"not equal y");

            Assert.IsTrue(x.Grad == null);
        }

        readonly float[] y_expect = {
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            0,
        };
    }
}
