using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection1D {
    [TestClass]
    public class ColumnToImageTest {
        [TestMethod]
        public void ExecuteTest() {
            int channels = 4, inwidth = 16, batch = 2;
            int kwidth = 3;
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[channels * inwidth * batch]).Select((_, idx) => idx * 2e-3f).ToArray();
            float[] yval = (new float[channels * inwidth * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

            ParameterField x = (Shape.Map1D(channels, inwidth, batch), xval);
            VariableField y_actual = (Shape.Map1D(channels, inwidth, batch), yval);

            Field y_expect = ColumnToImage1D(ImageToColumn1D(x, kwidth), kwidth);

            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();
        }
    }
}
