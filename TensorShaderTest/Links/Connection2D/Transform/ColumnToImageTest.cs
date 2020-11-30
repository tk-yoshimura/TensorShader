using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection2D {
    [TestClass]
    public class ColumnToImageTest {
        [TestMethod]
        public void ExecuteTest() {
            int channels = 4, inwidth = 16, inheight = 15, batch = 2;
            int kwidth = 3, kheight = 5;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[channels * inwidth * inheight * batch]).Select((_, idx) => idx * 2e-3f).ToArray();
            float[] yval = (new float[channels * inwidth * inheight * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

            ParameterField x = (Shape.Map2D(channels, inwidth, inheight, batch), xval);
            VariableField y_actual = (Shape.Map2D(channels, inwidth, inheight, batch), yval);

            Field y_expect = ColumnToImage2D(ImageToColumn2D(x, kwidth, kheight), kwidth, kheight);

            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();
        }
    }
}
