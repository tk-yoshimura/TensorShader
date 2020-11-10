using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Updaters.RestrictGrad;

namespace TensorShaderTest.Updaters.RestrictGrad {
    [TestClass]
    public class GradClippingTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 27;
            float limit = 1;

            float[] xval = (new float[length]).Select((_, idx) => 0.1f * ((float)idx * 3 - length)).ToArray();
            float[] yval = xval.Select((v) => Math.Min(Math.Max(v, -limit), limit)).ToArray();

            ParameterField x = xval;

            (Flow flow, _) = Flow.Optimize(x);
            flow.Execute();

            x.AddUpdater(new GradClipping(x, limit));
            x.Update();

            AssertError.Tolerance(yval, x.GradState, 1e-7f, 1e-5f);

            CollectionAssert.AreEqual(xval, x.State);
        }
    }
}
