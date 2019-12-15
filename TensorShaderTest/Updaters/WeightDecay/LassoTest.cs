using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Updaters.WeightDecay;

namespace TensorShaderTest.Updaters.WeightDecay {
    [TestClass]
    public class LassoTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 27;
            float decay = 0.25f;

            float[] xval = (new float[length]).Select((_, idx) => 0.1f * ((float)idx * 3 - length)).ToArray();
            float[] yval = xval.Select((v) => Math.Sign(v) * Math.Max(Math.Abs(v) - decay, 0)).ToArray();

            Tensor x_tensor = new Tensor(Shape.Vector(length), xval);

            ParameterField x = x_tensor;

            (Flow flow, _) = Flow.Optimize(x);

            x.AddUpdater(new Lasso(x, decay));
            x.Update();

            AssertError.Tolerance(yval, x_tensor.State, 1e-7f, 1e-5f);
        }
    }
}
