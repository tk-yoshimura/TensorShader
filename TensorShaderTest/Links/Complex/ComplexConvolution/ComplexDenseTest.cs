using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ComplexConvolution {
    [TestClass]
    public class ComplexDenseTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 6, outchannels = 8, batch = 3;

            float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map0D(inchannels, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map0D(outchannels, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel0D(inchannels, outchannels / 2), wval);

            ParameterField x = xtensor;
            ParameterField w = wtensor;
            VariableField y_actual = ytensor;

            Field y_expect = ComplexDense(x, w);
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
            int inchannels = 6, outchannels = 8, batch = 3;

            float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map0D(inchannels, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map0D(outchannels, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel0D(inchannels, outchannels / 2), wval);

            ParameterField x = xtensor;
            ParameterField w = wtensor;
            VariableField y_actual = ytensor;

            Field x_real = ComplexReal(x), x_imag = ComplexImag(x);
            Field w_real = ComplexReal(w), w_imag = ComplexImag(w);

            Field y_real = Dense(x_real, w_real) - Dense(x_imag, w_imag);
            Field y_imag = Dense(x_imag, w_real) + Dense(x_real, w_imag);

            Field y_expect = ComplexCast(y_real, y_imag);

            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;
            float[] gw_actual = w.GradState;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-9f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-9f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = new float[] {
            -2.471360e-04f,  -5.450800e-05f,  -1.921840e-04f,  -4.798800e-05f,  -1.372320e-04f,  -4.146800e-05f,
            -1.082120e-03f,  -5.778000e-05f,  -9.026240e-04f,  -5.442800e-05f,  -7.231280e-04f,  -5.107600e-05f,
            -1.917104e-03f,  -6.105200e-05f,  -1.613064e-03f,  -6.086800e-05f,  -1.309024e-03f,  -6.068400e-05f,
        };

        float[] gw_expect = new float[] {
            -4.949250e-04f,  3.395400e-05f,  -5.909550e-04f,  3.435600e-05f,  -6.869850e-04f,  3.475800e-05f,
            -5.816190e-04f,  3.147600e-05f,  -7.033770e-04f,  2.993400e-05f,  -8.251350e-04f,  2.839200e-05f,
            -6.683130e-04f,  2.899800e-05f,  -8.157990e-04f,  2.551200e-05f,  -9.632850e-04f,  2.202600e-05f,
            -7.550070e-04f,  2.652000e-05f,  -9.282210e-04f,  2.109000e-05f,  -1.101435e-03f,  1.566000e-05f,
        };
    }
}
