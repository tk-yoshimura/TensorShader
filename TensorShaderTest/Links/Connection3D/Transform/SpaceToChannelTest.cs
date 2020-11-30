using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection3D {
    [TestClass]
    public class SpaceToChannelTest {
        [TestMethod]
        public void ExecuteTest() {
            int scale = 3;
            int inchannels = 2, outwidth = 5, outheight = 4, outdepth = 3, batch = 2;
            int outchannels = inchannels * scale * scale * scale;
            int inwidth = outwidth * scale, inheight = outheight * scale, indepth = outdepth * scale;

            float[] xval = (new float[inchannels * inwidth * inheight * indepth * batch]).Select((_, idx) => idx * 2e-3f).ToArray();
            float[] yval = (new float[outchannels * outwidth * outheight * outdepth * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

            ParameterField x = (Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
            VariableField y_actual = (Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), yval);

            Field y_expect = SpaceToChannel3D(x, scale);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();
        }
    }
}
