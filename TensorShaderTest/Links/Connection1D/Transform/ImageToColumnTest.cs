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

            Tensor x1tensor = new Tensor(Shape.Map1D(channels, inwidth, batch), xval);
            Tensor x2tensor = new Tensor(Shape.Map1D(channels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(new Shape(ShapeType.Column, kwidth, channels, outwidth, batch), yval);

            ParameterField x1 = x1tensor;
            ParameterField x2 = x2tensor;
            VariableField y_actual = ytensor;

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

            (Flow flow, Parameters Parameters) = Flow.Optimize(err1, err2);

            flow.Execute();

            float[] gx1 = x1.GradTensor.State;
            float[] gx2 = x2.GradTensor.State;

            AssertError.Tolerance(gx1, gx2, 1e-7f, 1e-5f, $"not equal gx");
        }
    }
}