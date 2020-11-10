using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class CubeTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => ((float)idx - 12) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x = (Shape.Vector(length), xval);
            VariableField y_actual = (Shape.Vector(length), yval);

            Field y_expect = Cube(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -3.00000000e+00f,
            -2.04671827e+00f,
            -1.37924383e+00f,
            -9.22851562e-01f,
            -6.17283951e-01f,
            -4.15304302e-01f,
            -2.81250000e-01f,
            -1.89585745e-01f,
            -1.23456790e-01f,
            -7.32421875e-02f,
            -3.51080247e-02f,
            -9.56066744e-03f,
            -0.00000000e+00f,
            -1.12726659e-02f,
            -4.82253086e-02f,
            -1.14257812e-01f,
            -2.09876543e-01f,
            -3.31247589e-01f,
            -4.68750000e-01f,
            -6.05529032e-01f,
            -7.16049383e-01f,
            -7.64648438e-01f,
            -7.04089506e-01f,
            -4.74115066e-01f,
        };
    }
}
