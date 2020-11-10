using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Updaters.RestrictGrad;

namespace TensorShaderTest.Updaters.RestrictGrad {
    [TestClass]
    public class GradRMSClippingTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 27;

            {
                float limit = 0.5f;

                float[] xval = (new float[length]).Select((_, idx) => 0.1f * ((float)idx * 3 - length)).ToArray();
                float norm = xval.Select((v) => v * v).Average();
                float rate = (float)Math.Sqrt(limit * limit / Math.Max(norm, limit * limit));

                float[] yval = xval.Select((v) => v * rate).ToArray();

                ParameterField x = (Shape.Vector(length), xval);

                (Flow flow, _) = Flow.Optimize(x);
                flow.Execute();

                x.AddUpdater(new GradRMSClipping(x, limit));
                x.Update();

                AssertError.Tolerance(yval, x.GradState, 1e-7f, 1e-5f);

                float post_rms = (float)Math.Sqrt(x.GradState.Select((v) => v * v).Average());

                Assert.AreEqual(limit, post_rms, 1e-5f);

                CollectionAssert.AreEqual(xval, x.State);
            }

            {
                float limit = 1e+7f;

                float[] xval = (new float[length]).Select((_, idx) => 0.1f * ((float)idx * 3 - length)).ToArray();

                ParameterField x = (Shape.Vector(length), xval);

                (Flow flow, _) = Flow.Optimize(x);
                flow.Execute();

                x.AddUpdater(new GradRMSClipping(x, limit));
                x.Update();

                AssertError.Tolerance(xval, x.GradState, 1e-7f, 1e-5f);

                CollectionAssert.AreEqual(xval, x.State);
            }
        }
    }
}
