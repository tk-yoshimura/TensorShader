using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection2D {
    [TestClass]
    public class AveragePoolingTest {
        [TestMethod]
        public void ReferenceTest() {
            int stride = 3;
            int channels = 2, inwidth = 10, inheight = 9, batch = 2;
            int outwidth = inwidth / stride, outheight = inheight / stride;

            float[] xval = (new float[channels * inwidth * inheight * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[channels * outwidth * outheight * batch]).Select((_, idx) => idx * 2e-3f).ToArray();

            ParameterField x = (Shape.Map2D(channels, inwidth, inheight, batch), xval);
            VariableField y_actual = (Shape.Map2D(channels, outwidth, outheight, batch), yval);

            Field y_expect = AveragePooling2D(x, stride);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            2.44444444e-03f, 2.33333333e-03f, 2.44444444e-03f, 2.33333333e-03f, 2.44444444e-03f,
            2.33333333e-03f, 2.66666667e-03f, 2.55555556e-03f, 2.66666667e-03f, 2.55555556e-03f,
            2.66666667e-03f, 2.55555556e-03f, 2.88888889e-03f, 2.77777778e-03f, 2.88888889e-03f,
            2.77777778e-03f, 2.88888889e-03f, 2.77777778e-03f, 0.00000000e+00f, 0.00000000e+00f,
            2.44444444e-03f, 2.33333333e-03f, 2.44444444e-03f, 2.33333333e-03f, 2.44444444e-03f,
            2.33333333e-03f, 2.66666667e-03f, 2.55555556e-03f, 2.66666667e-03f, 2.55555556e-03f,
            2.66666667e-03f, 2.55555556e-03f, 2.88888889e-03f, 2.77777778e-03f, 2.88888889e-03f,
            2.77777778e-03f, 2.88888889e-03f, 2.77777778e-03f, 0.00000000e+00f, 0.00000000e+00f,
            2.44444444e-03f, 2.33333333e-03f, 2.44444444e-03f, 2.33333333e-03f, 2.44444444e-03f,
            2.33333333e-03f, 2.66666667e-03f, 2.55555556e-03f, 2.66666667e-03f, 2.55555556e-03f,
            2.66666667e-03f, 2.55555556e-03f, 2.88888889e-03f, 2.77777778e-03f, 2.88888889e-03f,
            2.77777778e-03f, 2.88888889e-03f, 2.77777778e-03f, 0.00000000e+00f, 0.00000000e+00f,
            7.77777778e-03f, 7.66666667e-03f, 7.77777778e-03f, 7.66666667e-03f, 7.77777778e-03f,
            7.66666667e-03f, 8.00000000e-03f, 7.88888889e-03f, 8.00000000e-03f, 7.88888889e-03f,
            8.00000000e-03f, 7.88888889e-03f, 8.22222222e-03f, 8.11111111e-03f, 8.22222222e-03f,
            8.11111111e-03f, 8.22222222e-03f, 8.11111111e-03f, 0.00000000e+00f, 0.00000000e+00f,
            7.77777778e-03f, 7.66666667e-03f, 7.77777778e-03f, 7.66666667e-03f, 7.77777778e-03f,
            7.66666667e-03f, 8.00000000e-03f, 7.88888889e-03f, 8.00000000e-03f, 7.88888889e-03f,
            8.00000000e-03f, 7.88888889e-03f, 8.22222222e-03f, 8.11111111e-03f, 8.22222222e-03f,
            8.11111111e-03f, 8.22222222e-03f, 8.11111111e-03f, 0.00000000e+00f, 0.00000000e+00f,
            7.77777778e-03f, 7.66666667e-03f, 7.77777778e-03f, 7.66666667e-03f, 7.77777778e-03f,
            7.66666667e-03f, 8.00000000e-03f, 7.88888889e-03f, 8.00000000e-03f, 7.88888889e-03f,
            8.00000000e-03f, 7.88888889e-03f, 8.22222222e-03f, 8.11111111e-03f, 8.22222222e-03f,
            8.11111111e-03f, 8.22222222e-03f, 8.11111111e-03f, 0.00000000e+00f, 0.00000000e+00f,
            1.31111111e-02f, 1.30000000e-02f, 1.31111111e-02f, 1.30000000e-02f, 1.31111111e-02f,
            1.30000000e-02f, 1.33333333e-02f, 1.32222222e-02f, 1.33333333e-02f, 1.32222222e-02f,
            1.33333333e-02f, 1.32222222e-02f, 1.35555556e-02f, 1.34444444e-02f, 1.35555556e-02f,
            1.34444444e-02f, 1.35555556e-02f, 1.34444444e-02f, 0.00000000e+00f, 0.00000000e+00f,
            1.31111111e-02f, 1.30000000e-02f, 1.31111111e-02f, 1.30000000e-02f, 1.31111111e-02f,
            1.30000000e-02f, 1.33333333e-02f, 1.32222222e-02f, 1.33333333e-02f, 1.32222222e-02f,
            1.33333333e-02f, 1.32222222e-02f, 1.35555556e-02f, 1.34444444e-02f, 1.35555556e-02f,
            1.34444444e-02f, 1.35555556e-02f, 1.34444444e-02f, 0.00000000e+00f, 0.00000000e+00f,
            1.31111111e-02f, 1.30000000e-02f, 1.31111111e-02f, 1.30000000e-02f, 1.31111111e-02f,
            1.30000000e-02f, 1.33333333e-02f, 1.32222222e-02f, 1.33333333e-02f, 1.32222222e-02f,
            1.33333333e-02f, 1.32222222e-02f, 1.35555556e-02f, 1.34444444e-02f, 1.35555556e-02f,
            1.34444444e-02f, 1.35555556e-02f, 1.34444444e-02f, 0.00000000e+00f, 0.00000000e+00f,
            1.84444444e-02f, 1.83333333e-02f, 1.84444444e-02f, 1.83333333e-02f, 1.84444444e-02f,
            1.83333333e-02f, 1.86666667e-02f, 1.85555556e-02f, 1.86666667e-02f, 1.85555556e-02f,
            1.86666667e-02f, 1.85555556e-02f, 1.88888889e-02f, 1.87777778e-02f, 1.88888889e-02f,
            1.87777778e-02f, 1.88888889e-02f, 1.87777778e-02f, 0.00000000e+00f, 0.00000000e+00f,
            1.84444444e-02f, 1.83333333e-02f, 1.84444444e-02f, 1.83333333e-02f, 1.84444444e-02f,
            1.83333333e-02f, 1.86666667e-02f, 1.85555556e-02f, 1.86666667e-02f, 1.85555556e-02f,
            1.86666667e-02f, 1.85555556e-02f, 1.88888889e-02f, 1.87777778e-02f, 1.88888889e-02f,
            1.87777778e-02f, 1.88888889e-02f, 1.87777778e-02f, 0.00000000e+00f, 0.00000000e+00f,
            1.84444444e-02f, 1.83333333e-02f, 1.84444444e-02f, 1.83333333e-02f, 1.84444444e-02f,
            1.83333333e-02f, 1.86666667e-02f, 1.85555556e-02f, 1.86666667e-02f, 1.85555556e-02f,
            1.86666667e-02f, 1.85555556e-02f, 1.88888889e-02f, 1.87777778e-02f, 1.88888889e-02f,
            1.87777778e-02f, 1.88888889e-02f, 1.87777778e-02f, 0.00000000e+00f, 0.00000000e+00f,
            2.37777778e-02f, 2.36666667e-02f, 2.37777778e-02f, 2.36666667e-02f, 2.37777778e-02f,
            2.36666667e-02f, 2.40000000e-02f, 2.38888889e-02f, 2.40000000e-02f, 2.38888889e-02f,
            2.40000000e-02f, 2.38888889e-02f, 2.42222222e-02f, 2.41111111e-02f, 2.42222222e-02f,
            2.41111111e-02f, 2.42222222e-02f, 2.41111111e-02f, 0.00000000e+00f, 0.00000000e+00f,
            2.37777778e-02f, 2.36666667e-02f, 2.37777778e-02f, 2.36666667e-02f, 2.37777778e-02f,
            2.36666667e-02f, 2.40000000e-02f, 2.38888889e-02f, 2.40000000e-02f, 2.38888889e-02f,
            2.40000000e-02f, 2.38888889e-02f, 2.42222222e-02f, 2.41111111e-02f, 2.42222222e-02f,
            2.41111111e-02f, 2.42222222e-02f, 2.41111111e-02f, 0.00000000e+00f, 0.00000000e+00f,
            2.37777778e-02f, 2.36666667e-02f, 2.37777778e-02f, 2.36666667e-02f, 2.37777778e-02f,
            2.36666667e-02f, 2.40000000e-02f, 2.38888889e-02f, 2.40000000e-02f, 2.38888889e-02f,
            2.40000000e-02f, 2.38888889e-02f, 2.42222222e-02f, 2.41111111e-02f, 2.42222222e-02f,
            2.41111111e-02f, 2.42222222e-02f, 2.41111111e-02f, 0.00000000e+00f, 0.00000000e+00f,
            2.91111111e-02f, 2.90000000e-02f, 2.91111111e-02f, 2.90000000e-02f, 2.91111111e-02f,
            2.90000000e-02f, 2.93333333e-02f, 2.92222222e-02f, 2.93333333e-02f, 2.92222222e-02f,
            2.93333333e-02f, 2.92222222e-02f, 2.95555556e-02f, 2.94444444e-02f, 2.95555556e-02f,
            2.94444444e-02f, 2.95555556e-02f, 2.94444444e-02f, 0.00000000e+00f, 0.00000000e+00f,
            2.91111111e-02f, 2.90000000e-02f, 2.91111111e-02f, 2.90000000e-02f, 2.91111111e-02f,
            2.90000000e-02f, 2.93333333e-02f, 2.92222222e-02f, 2.93333333e-02f, 2.92222222e-02f,
            2.93333333e-02f, 2.92222222e-02f, 2.95555556e-02f, 2.94444444e-02f, 2.95555556e-02f,
            2.94444444e-02f, 2.95555556e-02f, 2.94444444e-02f, 0.00000000e+00f, 0.00000000e+00f,
            2.91111111e-02f, 2.90000000e-02f, 2.91111111e-02f, 2.90000000e-02f, 2.91111111e-02f,
            2.90000000e-02f, 2.93333333e-02f, 2.92222222e-02f, 2.93333333e-02f, 2.92222222e-02f,
            2.93333333e-02f, 2.92222222e-02f, 2.95555556e-02f, 2.94444444e-02f, 2.95555556e-02f,
            2.94444444e-02f, 2.95555556e-02f, 2.94444444e-02f, 0.00000000e+00f, 0.00000000e+00f,
        };
    }
}
