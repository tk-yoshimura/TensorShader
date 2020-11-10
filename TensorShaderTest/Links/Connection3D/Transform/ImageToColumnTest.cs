using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection3D {
    [TestClass]
    public class ImageToColumnTest {
        [TestMethod]
        public void ReferenceTest() {
            int channels = 4, inwidth = 16, inheight = 15, indepth = 14, batch = 2;
            int kwidth = 3, kheight = 5, kdepth = 7;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[channels * inwidth * inheight * indepth * batch]).Select((_, idx) => idx * 2e-3f).ToArray();
            float[] yval = (new float[kwidth * kheight * kdepth * channels * outwidth * outheight * outdepth * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

            ParameterField x1 = (Shape.Map3D(channels, inwidth, inheight, indepth, batch), xval);
            ParameterField x2 = (Shape.Map3D(channels, inwidth, inheight, indepth, batch), xval);
            VariableField y_actual = (new Shape(ShapeType.Column, kwidth * kheight * kdepth, channels, outwidth, outheight, outdepth, batch), yval);

            Field y1_expect = ImageToColumn3D(x1, kwidth, kheight, kdepth);

            List<Field> x2s = new List<Field>();
            for (int kz = 0; kz < kdepth; kz++) {
                for (int ky = 0; ky < kheight; ky++) {
                    for (int kx = 0; kx < kwidth; kx++) {
                        x2s.Add(
                            Reshape(
                                Trimming3D(x2, kx, kwidth - kx - 1, ky, kheight - ky - 1, kz, kdepth - kz - 1),
                                    new Shape(ShapeType.Column, 1, channels, outwidth, outheight, outdepth, batch)
                                )
                            );
                    }
                }
            }

            Field y2_expect = Concat(Axis.Column.Filter, x2s.ToArray());

            Field err1 = y1_expect - y_actual;
            Field err2 = y2_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err1, err2);

            flow.Execute();

            float[] gx1 = x1.GradState;
            float[] gx2 = x2.GradState;

            AssertError.Tolerance(gx1, gx2, 1e-7f, 1e-5f, $"not equal gx");
        }
    }
}
