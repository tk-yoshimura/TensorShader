using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection1D {
    [TestClass]
    public class ChannelToSpaceTest {
        [TestMethod]
        public void ReferenceTest() {
            int scale = 3;
            int outchannels = 2, inwidth = 4, batch = 2;
            int inchannels = outchannels * scale, outwidth = inwidth * scale;

            float[] xval = (new float[inchannels * inwidth * batch]).Select((_, idx) => idx * 2e-3f).ToArray();
            float[] yval = (new float[outchannels * outwidth * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

            ParameterField x = (Shape.Map1D(inchannels, inwidth, batch), xval);
            VariableField y_actual = (Shape.Map1D(outchannels, outwidth, batch), yval);

            Field y_expect = ChannelToSpace1D(x, scale);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();
        }
    }
}
