using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection1D {
    [TestClass]
    public class ImageToColumnTest {
        [TestMethod]
        public void ReferenceTest() {
            int channels = 4, inwidth = 16, batch = 2;
            int kwidth = 3;
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[channels * inwidth * batch]).Select((_, idx) => idx * 2e-3f).ToArray();
            float[] yval = (new float[kwidth * channels * outwidth * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

            ParameterField x1 = (Shape.Map1D(channels, inwidth, batch), xval);
            ParameterField x2 = (Shape.Map1D(channels, inwidth, batch), xval);
            VariableField y_actual = (new Shape(ShapeType.Column, kwidth, channels, outwidth, batch), yval);

            Field y1_expect = ImageToColumn1D(x1, kwidth);

            List<Field> x2s = new List<Field>();
            for (int kx = 0; kx < kwidth; kx++) {
                x2s.Add(
                    Reshape(
                        Trimming1D(x2, kx, kwidth - kx - 1),
                        new Shape(ShapeType.Column, 1, channels, outwidth, batch)
                        )
                    );
            }

            Field y2_expect = Concat(Axis.Column.Filter, x2s.ToArray());

            Field err1 = y1_expect - y_actual;
            Field err2 = y2_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err1, err2);

            flow.Execute();

            float[] gx1 = x1.GradState.Value;
            float[] gx2 = x2.GradState.Value;

            AssertError.Tolerance(gx1, gx2, 1e-7f, 1e-5f, $"not equal gx");
        }
    }
}
