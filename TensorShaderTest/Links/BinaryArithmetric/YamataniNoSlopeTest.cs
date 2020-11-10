using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.BinaryArithmetric {
    [TestClass]
    public class YamataniNoSlopeTest {
        const float slope = 0.0f;

        [TestMethod]
        public void ReferenceTest() {
            int length = 128;

            float[] x1val = (new float[length]).Select((_, idx) => (float)Math.Sin(idx * 0.1) * idx * 0.1f).ToArray();
            float[] x2val = (new float[length]).Select((_, idx) => (float)Math.Cos(idx * 0.1) * idx * 0.1f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x1 = (Shape.Vector(length), x1val);
            ParameterField x2 = (Shape.Vector(length), x2val);
            VariableField y_actual = (Shape.Vector(length), yval);

            Field y_expect = Yamatani(x1, x2, slope);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx1_actual = x1.GradState;
            float[] gx2_actual = x2.GradState;

            AssertError.Tolerance(gx1_expect, gx1_actual, 1e-7f, 1e-5f, $"not equal gx1");

            AssertError.Tolerance(gx2_expect, gx2_actual, 1e-7f, 1e-5f, $"not equal gx2");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int length = 128;

            float[] x1val = (new float[length]).Select((_, idx) => (float)Math.Sin(idx * 0.1) * idx * 0.1f).ToArray();
            float[] x2val = (new float[length]).Select((_, idx) => (float)Math.Cos(idx * 0.1) * idx * 0.1f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x1 = (Shape.Vector(length), x1val);
            ParameterField x2 = (Shape.Vector(length), x2val);
            VariableField y_actual = (Shape.Vector(length), yval);

            Field s = slope * (x1 + x2);
            Field pp = Minimum(x1, x2) * GreaterThan(x1, 0) * GreaterThan(x2, 0);
            Field nn = Maximum(x1, x2) * LessThan(x1, 0) * LessThan(x2, 0);

            StoreField y_expect = s + pp + nn;
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx1_actual = x1.GradState;
            float[] gx2_actual = x2.GradState;

            AssertError.Tolerance(gx1_expect, gx1_actual, 1e-7f, 1e-5f, $"not equal gx1");

            AssertError.Tolerance(gx2_expect, gx2_actual, 1e-7f, 1e-5f, $"not equal gx2");
        }

        float[] gx1_expect = {
            0.00000000e+00f,  -3.16833250e-02f,  -4.35994672e-02f,  -3.63439380e-02f,  -1.08993297e-02f,  3.13794360e-02f,  8.87854840e-02f,  1.59285714e-01f, 
            0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f, 
            -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f, 
            -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f, 
            -1.52013059e+00f,  -1.89556079e+00f,  -2.28550641e+00f,  -2.68607463e+00f,  -3.09307360e+00f,  -3.50206039e+00f,  -3.90839332e+00f,  -4.30728802e+00f, 
            0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f, 
            0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f, 
            0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  -2.51907243e+00f, 
            -1.92075176e+00f,  -1.31005341e+00f,  -6.93827001e-01f,  -7.91721985e-02f,  5.26637454e-01f,  1.11623437e+00f,  1.68223952e+00f,  0.00000000e+00f, 
            0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f, 
            -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f, 
            -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -4.67226898e+00f, 
            -5.67353710e+00f,  -6.67774474e+00f,  -7.67482880e+00f,  -8.65460535e+00f,  -9.60687778e+00f,  -1.05215469e+01f,  -1.13887218e+01f,  0.00000000e+00f, 
            0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f, 
            0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f, 
            0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  -4.82634961e+00f,  -3.59961974e+00f, 
        };

        float[] gx2_expect = {
            0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f, 
            2.24032034e-01f,  1.84448971e-01f,  1.23635639e-01f,  4.06224002e-02f,  -6.51706946e-02f,  -1.93918189e-01f,  -3.45379333e-01f,  -5.18894197e-01f, 
            0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f, 
            0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f, 
            0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f, 
            -4.28124115e+00f,  -4.06511151e+00f,  -3.80909545e+00f,  -3.51510311e+00f,  -3.18559796e+00f,  -2.82358110e+00f,  -2.43256829e+00f,  -2.01656005e+00f, 
            -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f, 
            -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f, 
            -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  1.90194800e+00f, 
            1.38012946e+00f,  7.98699210e-01f,  1.61916891e-01f,  -5.25235116e-01f,  -1.25709186e+00f,  -2.02735460e+00f,  -2.82914772e+00f,  0.00000000e+00f, 
            0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f, 
            0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f, 
            0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  -1.08921788e+01f, 
            -1.01675696e+01f,  -9.36813774e+00f,  -8.50125149e+00f,  -7.57512865e+00f,  -6.59876299e+00f,  -5.58184114e+00f,  -0.00000000e+00f,  -0.00000000e+00f, 
            -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f, 
            -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f,  -0.00000000e+00f, 
        };
    }
}
