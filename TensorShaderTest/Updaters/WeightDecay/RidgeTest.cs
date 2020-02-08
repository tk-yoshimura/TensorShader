using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Updaters.WeightDecay;

namespace TensorShaderTest.Updaters.WeightDecay {
    [TestClass]
    public class RidgeTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 27;
            float decay = 0.25f;

            float[] xval = (new float[length]).Select((_, idx) => 0.1f * ((float)idx * 3 - length)).ToArray();
            float[] yval = xval.Select((v) => v * (1 - decay)).ToArray();

            ParameterField x = new Tensor(Shape.Vector(length), xval);

            (Flow flow, _) = Flow.Optimize(x);

            x.AddUpdater(new Ridge(x, decay));
            x.Update();

            AssertError.Tolerance(yval, x.State, 1e-7f, 1e-5f);
        }
    }
}
