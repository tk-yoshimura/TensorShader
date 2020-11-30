using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection3D {
    [TestClass]
    public class ColumnToImageTest {
        [TestMethod]
        public void ExecuteTest() {
            int channels = 4, inwidth = 16, inheight = 15, indepth = 14, batch = 2;
            int kwidth = 3, kheight = 5, kdepth = 7;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[channels * inwidth * inheight * indepth * batch]).Select((_, idx) => idx * 2e-3f).ToArray();
            float[] yval = (new float[channels * inwidth * inheight * indepth * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

            ParameterField x = (Shape.Map3D(channels, inwidth, inheight, indepth, batch), xval);
            VariableField y_actual = (Shape.Map3D(channels, inwidth, inheight, indepth, batch), yval);

            Field y_expect = ColumnToImage3D(ImageToColumn3D(x, kwidth, kheight, kdepth), kwidth, kheight, kdepth);

            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();
        }
    }
}
