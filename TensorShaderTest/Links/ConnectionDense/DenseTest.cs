using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ConnectionDense {
    [TestClass]
    public class DenseTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, batch = 2;

            float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            ParameterField x = (Shape.Map0D(inchannels, batch), xval);
            ParameterField w = (Shape.Kernel0D(inchannels, outchannels), wval);
            VariableField y_actual = (Shape.Map0D(outchannels, batch), yval);

            Field y_expect = Dense(x, w);
            Field err = Abs(y_expect - y_actual);

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;
            float[] gw_actual = w.GradState.Value;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        readonly float[] gx_expect = new float[] {
            -1.024540e-03f, -9.780100e-04f, -9.314800e-04f, -8.849500e-04f, -8.384200e-04f,
            -7.918900e-04f, -7.453600e-04f, -4.881668e-03f, -4.734620e-03f, -4.587572e-03f,
            -4.440524e-03f, -4.293476e-03f, -4.146428e-03f, -3.999380e-03f
        };
        readonly float[] gw_expect = new float[] {
            -4.142600e-05f, -4.583900e-05f, -5.025200e-05f, -5.466500e-05f, -5.907800e-05f,
            -6.349100e-05f, -6.790400e-05f, -5.185600e-05f, -5.890600e-05f, -6.595600e-05f,
            -7.300600e-05f, -8.005600e-05f, -8.710600e-05f, -9.415600e-05f, -6.228600e-05f,
            -7.197300e-05f, -8.166000e-05f, -9.134700e-05f, -1.010340e-04f, -1.107210e-04f,
            -1.204080e-04f, -7.271600e-05f, -8.504000e-05f, -9.736400e-05f, -1.096880e-04f,
            -1.220120e-04f, -1.343360e-04f, -1.466600e-04f, -8.314600e-05f, -9.810700e-05f,
            -1.130680e-04f, -1.280290e-04f, -1.429900e-04f, -1.579510e-04f, -1.729120e-04f,
            -9.357600e-05f, -1.111740e-04f, -1.287720e-04f, -1.463700e-04f, -1.639680e-04f,
            -1.815660e-04f, -1.991640e-04f, -1.040060e-04f, -1.242410e-04f, -1.444760e-04f,
            -1.647110e-04f, -1.849460e-04f, -2.051810e-04f, -2.254160e-04f, -1.144360e-04f,
            -1.373080e-04f, -1.601800e-04f, -1.830520e-04f, -2.059240e-04f, -2.287960e-04f,
            -2.516680e-04f, -1.248660e-04f, -1.503750e-04f, -1.758840e-04f, -2.013930e-04f,
            -2.269020e-04f, -2.524110e-04f, -2.779200e-04f, -1.352960e-04f, -1.634420e-04f,
            -1.915880e-04f, -2.197340e-04f, -2.478800e-04f, -2.760260e-04f, -3.041720e-04f,
            -1.457260e-04f, -1.765090e-04f, -2.072920e-04f, -2.380750e-04f, -2.688580e-04f,
            -2.996410e-04f, -3.304240e-04f
        };
    }
}
