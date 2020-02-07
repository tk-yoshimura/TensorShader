using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection3D {
    [TestClass]
    public class LinearZoomTest {
        [TestMethod]
        public void ReferenceTest() {
            int scale = 2;
            int channels = 2, inwidth = 5, inheight = 4, indepth = 3, batch = 2;
            int outwidth = inwidth * scale, outheight = inheight * scale, outdepth = indepth * scale;

            float[] xval = (new float[channels * inwidth * inheight * indepth * batch]).Select((_, idx) => idx * 2e-3f).ToArray();
            float[] yval = (new float[channels * outwidth * outheight * outdepth * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

            Tensor xtensor = new Tensor(Shape.Map3D(channels, inwidth, inheight, indepth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map3D(channels, outwidth, outheight, outdepth, batch), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = LinearZoom3D(x);
            Field err = Abs(y_expect - y_actual);

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -7.366667e-02f, -7.266667e-02f, -7.433333e-02f, -7.333333e-02f, -7.433333e-02f, -7.333333e-02f, -7.433333e-02f, -7.333333e-02f,
            -7.500000e-02f, -7.400000e-02f, -9.700000e-02f, -9.600000e-02f, -9.766667e-02f, -9.666667e-02f, -9.766667e-02f, -9.666667e-02f,
            -9.766667e-02f, -9.666667e-02f, -9.833333e-02f, -9.733333e-02f, -1.170000e-01f, -1.160000e-01f, -1.176667e-01f, -1.166667e-01f,
            -1.176667e-01f, -1.166667e-01f, -1.176667e-01f, -1.166667e-01f, -1.183333e-01f, -1.173333e-01f, -1.403333e-01f, -1.393333e-01f,
            -1.410000e-01f, -1.400000e-01f, -1.410000e-01f, -1.400000e-01f, -1.410000e-01f, -1.400000e-01f, -1.416667e-01f, -1.406667e-01f,
            -3.270000e-01f, -3.260000e-01f, -3.276667e-01f, -3.266667e-01f, -3.276667e-01f, -3.266667e-01f, -3.276667e-01f, -3.266667e-01f,
            -3.283333e-01f, -3.273333e-01f, -3.503333e-01f, -3.493333e-01f, -3.510000e-01f, -3.500000e-01f, -3.510000e-01f, -3.500000e-01f,
            -3.510000e-01f, -3.500000e-01f, -3.516667e-01f, -3.506667e-01f, -3.703333e-01f, -3.693333e-01f, -3.710000e-01f, -3.700000e-01f,
            -3.710000e-01f, -3.700000e-01f, -3.710000e-01f, -3.700000e-01f, -3.716667e-01f, -3.706667e-01f, -3.936667e-01f, -3.926667e-01f,
            -3.943333e-01f, -3.933333e-01f, -3.943333e-01f, -3.933333e-01f, -3.943333e-01f, -3.933333e-01f, -3.950000e-01f, -3.940000e-01f,
            -5.803333e-01f, -5.793333e-01f, -5.810000e-01f, -5.800000e-01f, -5.810000e-01f, -5.800000e-01f, -5.810000e-01f, -5.800000e-01f,
            -5.816667e-01f, -5.806667e-01f, -6.036667e-01f, -6.026667e-01f, -6.043333e-01f, -6.033333e-01f, -6.043333e-01f, -6.033333e-01f,
            -6.043333e-01f, -6.033333e-01f, -6.050000e-01f, -6.040000e-01f, -6.236667e-01f, -6.226667e-01f, -6.243333e-01f, -6.233333e-01f,
            -6.243333e-01f, -6.233333e-01f, -6.243333e-01f, -6.233333e-01f, -6.250000e-01f, -6.240000e-01f, -6.470000e-01f, -6.460000e-01f,
            -6.476667e-01f, -6.466667e-01f, -6.476667e-01f, -6.466667e-01f, -6.476667e-01f, -6.466667e-01f, -6.483333e-01f, -6.473333e-01f,
            -7.936667e-01f, -7.926667e-01f, -7.943333e-01f, -7.933333e-01f, -7.943333e-01f, -7.933333e-01f, -7.943333e-01f, -7.933333e-01f,
            -7.950000e-01f, -7.940000e-01f, -8.170000e-01f, -8.160000e-01f, -8.176667e-01f, -8.166667e-01f, -8.176667e-01f, -8.166667e-01f,
            -8.176667e-01f, -8.166667e-01f, -8.183333e-01f, -8.173333e-01f, -8.370000e-01f, -8.360000e-01f, -8.376667e-01f, -8.366667e-01f,
            -8.376667e-01f, -8.366667e-01f, -8.376667e-01f, -8.366667e-01f, -8.383333e-01f, -8.373333e-01f, -8.603333e-01f, -8.593333e-01f,
            -8.610000e-01f, -8.600000e-01f, -8.610000e-01f, -8.600000e-01f, -8.610000e-01f, -8.600000e-01f, -8.616667e-01f, -8.606667e-01f,
            -1.047000e+00f, -1.046000e+00f, -1.047667e+00f, -1.046667e+00f, -1.047667e+00f, -1.046667e+00f, -1.047667e+00f, -1.046667e+00f,
            -1.048333e+00f, -1.047333e+00f, -1.070333e+00f, -1.069333e+00f, -1.071000e+00f, -1.070000e+00f, -1.071000e+00f, -1.070000e+00f,
            -1.071000e+00f, -1.070000e+00f, -1.071667e+00f, -1.070667e+00f, -1.090333e+00f, -1.089333e+00f, -1.091000e+00f, -1.090000e+00f,
            -1.091000e+00f, -1.090000e+00f, -1.091000e+00f, -1.090000e+00f, -1.091667e+00f, -1.090667e+00f, -1.113667e+00f, -1.112667e+00f,
            -1.114333e+00f, -1.113333e+00f, -1.114333e+00f, -1.113333e+00f, -1.114333e+00f, -1.113333e+00f, -1.115000e+00f, -1.114000e+00f,
            -1.300333e+00f, -1.299333e+00f, -1.301000e+00f, -1.300000e+00f, -1.301000e+00f, -1.300000e+00f, -1.301000e+00f, -1.300000e+00f,
            -1.301667e+00f, -1.300667e+00f, -1.323667e+00f, -1.322667e+00f, -1.324333e+00f, -1.323333e+00f, -1.324333e+00f, -1.323333e+00f,
            -1.324333e+00f, -1.323333e+00f, -1.325000e+00f, -1.324000e+00f, -1.343667e+00f, -1.342667e+00f, -1.344333e+00f, -1.343333e+00f,
            -1.344333e+00f, -1.343333e+00f, -1.344333e+00f, -1.343333e+00f, -1.345000e+00f, -1.344000e+00f, -1.367000e+00f, -1.366000e+00f,
            -1.367667e+00f, -1.366667e+00f, -1.367667e+00f, -1.366667e+00f, -1.367667e+00f, -1.366667e+00f, -1.368333e+00f, -1.367333e+00f,
        };
    }
}
