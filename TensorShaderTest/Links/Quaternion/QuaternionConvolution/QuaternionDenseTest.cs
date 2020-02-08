using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.QuaternionConvolution {
    [TestClass]
    public class QuaternionDenseTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 8, outchannels = 12, batch = 3;

            float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            ParameterField x = new Tensor(Shape.Map0D(inchannels, batch), xval);
            ParameterField w = new Tensor(Shape.Kernel0D(inchannels, outchannels / 4), wval);
            VariableField y_actual = new Tensor(Shape.Map0D(outchannels, batch), yval);

            Field y_expect = QuaternionDense(x, w);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;
            float[] gw_actual = w.GradState;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int inchannels = 8, outchannels = 12, batch = 3;

            float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            ParameterField x = new Tensor(Shape.Map0D(inchannels, batch), xval);
            ParameterField w = new Tensor(Shape.Kernel0D(inchannels, outchannels / 4), wval);
            VariableField y_actual = new Tensor(Shape.Map0D(outchannels, batch), yval);

            Field xr = QuaternionR(x), xi = QuaternionI(x), xj = QuaternionJ(x), xk = QuaternionK(x);
            Field wr = QuaternionR(w), wi = QuaternionI(w), wj = QuaternionJ(w), wk = QuaternionK(w);

            Field yr = Dense(xr, wr) - Dense(xi, wi) - Dense(xj, wj) - Dense(xk, wk);
            Field yi = Dense(xr, wi) + Dense(xi, wr) + Dense(xj, wk) - Dense(xk, wj);
            Field yj = Dense(xr, wj) - Dense(xi, wk) + Dense(xj, wr) + Dense(xk, wi);
            Field yk = Dense(xr, wk) + Dense(xi, wj) - Dense(xj, wi) + Dense(xk, wr);

            Field y_expect = QuaternionCast(yr, yi, yj, yk);

            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;
            float[] gw_actual = w.GradState;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = new float[] {
            -6.112800e-04f,  -1.013400e-04f,  1.789600e-05f,  -2.054440e-04f,  -3.497280e-04f,  -7.978800e-05f,
            1.381600e-05f,  -1.625800e-04f,  -2.516464e-03f,  -1.356760e-04f,  5.632800e-05f,  -3.113960e-04f,
            -1.688896e-03f,  -1.218040e-04f,  4.303200e-05f,  -2.769800e-04f,  -4.421648e-03f,  -1.700120e-04f,
            9.476000e-05f,  -4.173480e-04f,  -3.028064e-03f,  -1.638200e-04f,  7.224800e-05f,  -3.913800e-04f,
        };

        float[] gw_expect = new float[] {
            -2.244516e-03f,  7.224800e-05f,  1.155880e-04f,  8.966400e-05f,  -2.872404e-03f,  9.476000e-05f,
            9.115600e-05f,  8.424000e-05f,  -2.730500e-03f,  4.303200e-05f,  1.347560e-04f,  8.406400e-05f,
            -3.558068e-03f,  5.632800e-05f,  1.003400e-04f,  7.019200e-05f,  -3.216484e-03f,  1.381600e-05f,
            1.539240e-04f,  7.846400e-05f,  -4.243732e-03f,  1.789600e-05f,  1.095240e-04f,  5.614400e-05f,
        };
    }
}
