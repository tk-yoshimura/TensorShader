using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.BinaryArithmetric {
    [TestClass]
    public class Arctan2Test {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] x1val = (new float[length]).Select((_, idx) => (float)Math.Sin(idx)).ToArray();
            float[] x2val = (new float[length]).Select((_, idx) => (float)Math.Cos(idx)).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x1 = (Shape.Vector(length), x1val);
            ParameterField x2 = (Shape.Vector(length), x2val);
            VariableField y_actual = (Shape.Vector(length), yval);

            Field y_expect = Arctan2(x1, x2);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx1_actual = x1.GradState;
            float[] gx2_actual = x2.GradState;

            AssertError.Tolerance(gx1_expect, gx1_actual, 1e-6f, 1e-4f, $"not equal gx1"); /*nonlinear tolerance*/
            AssertError.Tolerance(gx2_expect, gx2_actual, 1e-6f, 1e-4f, $"not equal gx2"); /*nonlinear tolerance*/
        }

        float[] gx1_expect = {
            0.00000000e+00f,
            5.17789710e-01f,
            -7.97614770e-01f,
            -2.84622843e+00f,
            1.60133011e+00f,
            -4.23087437e-01f,
            -5.11948689e-01f,
            3.20520055e-01f,
            -2.01296585e-01f,
            -2.13369823e+00f,
            2.50298165e+00f,
            -8.96072819e-03f,
            -8.99861064e-01f,
            -9.80380829e-02f,
            1.16267117e-01f,
            -1.37399388e+00f,
            3.36734390e+00f,
            7.03837346e-01f,
            -1.05621350e+00f,
            -6.33979734e-01f,
            1.29407207e-01f,
            -6.98598042e-01f,
            4.04924927e+00f,
            1.64702659e+00f,
        };

        float[] gx2_expect = {
            -0.00000000e+00f,
            -8.06409694e-01f,
            -1.74282007e+00f,
            -4.05720023e-01f,
            -1.85405409e+00f,
            -1.43025343e+00f,
            -1.48980238e-01f,
            -2.79316556e-01f,
            -1.36875869e+00f,
            -9.65105124e-01f,
            -1.62283526e+00f,
            -2.02468412e+00f,
            -5.72185592e-01f,
            4.53937042e-02f,
            -8.42309524e-01f,
            -1.17612970e+00f,
            -1.01233215e+00f,
            -2.45914831e+00f,
            -1.20124610e+00f,
            9.61046523e-02f,
            -2.89504749e-01f,
            -1.06710748e+00f,
            -3.58425617e-02f,
            -2.61573037e+00f,
        };
    }
}
