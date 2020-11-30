using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Updaters.OptimizeMethod;
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

            ParameterField x = xval;

            (Flow flow, Parameters parameters) = Flow.Optimize(x);

            parameters.AddUpdater((parameters) => new Lasso(x, decay))
                      .AddUpdater((parameters) => new SGD(parameters, 1));
            parameters.Update();

            AssertError.Tolerance(yval, x.State.Value, 1e-7f, 1e-5f);
        }

        [TestMethod]
        public void DependGradTest() {
            int length = 27;
            float decay = 0.25f;

            float[] xval = (new float[length]).Select((_, idx) => 0.1f * ((float)idx * 3 - length)).ToArray();

            ParameterField x = xval;

            (Flow flow, Parameters parameters) = Flow.Optimize(x);

            parameters.AddUpdater((parameters) => new Lasso(x, decay, depend_grad: true))
                      .AddUpdater((parameters) => new SGD(parameters, 1));
            parameters.Update();

            AssertError.Tolerance(xval, x.State.Value, 1e-7f, 1e-5f);
        }

        [TestMethod]
        public void DependGrad2Test() {
            int length = 27;
            float decay = 0.05f;

            float[] xval = (new float[length]).Select((_, idx) => 0.1f * ((float)idx * 3 - length)).ToArray();
            float[] tval = (new float[length]).Select((_, idx) => 0.2f * ((float)idx * 2 - length)).ToArray();

            float absmean = xval.Select((_, i) => Math.Abs(xval[i] - tval[i])).Average();

            float[] yval = xval.Select((v, i) => tval[i] - Math.Sign(v) * absmean * decay).ToArray();

            ParameterField x = xval;
            VariableField t = tval;

            (Flow flow, Parameters parameters) = Flow.Optimize(x - t);
            parameters.AddUpdater((parameters) => new Lasso(x, decay, depend_grad: true))
                      .AddUpdater((parameters) => new SGD(parameters, 1));

            flow.Execute();

            parameters.Update();

            AssertError.Tolerance(yval, x.State.Value, 1e-7f, 1e-5f);
        }
    }
}
