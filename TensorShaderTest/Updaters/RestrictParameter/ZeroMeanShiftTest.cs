using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Updaters.RestrictParameter;
using static TensorShader.Tensor;

namespace TensorShaderTest.Updaters.RestrictParameter {
    [TestClass]
    public class ZeroMeanShiftTest {
        [TestMethod]
        public void ReferenceTest() {
            int ch = 7, w = 3, h = 5, length = ch * w * h;
            float decay = 0.25f;

            float[] xval = (new float[length]).Select((_, idx) => 0.1f * ((float)idx * 3 * (idx % 5) - length)).ToArray();
            float[] yval = xval.Select((v) => Math.Sign(v) * Math.Max(Math.Abs(v) - decay, 0)).ToArray();

            Tensor x_tensor = (Shape.Map2D(ch, w, h), xval);

            int[] axes = new int[] { Axis.Map2D.Width, Axis.Map2D.Height };

            float[] pre_variance = Variance(x_tensor, axes).State;
            float[] pre_mean = Average(x_tensor, axes).State;

            ParameterField x = x_tensor;

            (Flow flow, _) = Flow.Optimize(x);

            x.AddUpdater(new ZeroMeanShift(x, axes));
            x.Update();

            float[] post_variance = Variance(x_tensor, axes).State;
            float[] post_mean = Average(x_tensor, axes).State;

            AssertError.Tolerance(pre_variance, post_variance, 1e-7f, 1e-5f);

            foreach (var v in pre_mean) {
                Assert.AreNotEqual(0, v, 1e-5f);
            }

            foreach (var v in post_mean) {
                Assert.AreEqual(0, v, 1e-5f);
            }
        }
    }
}
