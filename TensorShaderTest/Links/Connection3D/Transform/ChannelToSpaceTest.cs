using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection3D {
    [TestClass]
    public class ChannelToSpaceTest {
        [TestMethod]
        public void ExecuteTest() {
            int scale = 3;
            int outchannels = 2, inwidth = 5, inheight = 4, indepth = 3, batch = 2;
            int inchannels = outchannels * scale * scale * scale;
            int outwidth = inwidth * scale, outheight = inheight * scale, outdepth = indepth * scale;

            float[] xval = (new float[inchannels * inwidth * inheight * indepth * batch]).Select((_, idx) => idx * 2e-3f).ToArray();
            float[] yval = (new float[outchannels * outwidth * outheight * outdepth * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

            Tensor xtensor = new Tensor(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = ChannelToSpace3D(x, scale);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();
        }
    }
}
