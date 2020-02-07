using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Complex {
    [TestClass]
    public class ComplexCastTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length / 2]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length / 2]).Select((_, idx) => (float)idx * 3 - length).Reverse().ToArray();

            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length / 2), xval);
            Tensor ytensor = new Tensor(Shape.Vector(length / 2), yval);

            Tensor ttensor = new Tensor(Shape.Vector(length), tval);

            ParameterField x = xtensor;
            ParameterField y = ytensor;
            VariableField t_actual = ttensor;

            Field t_expect = ComplexCast(x, y);
            StoreField err = t_expect - t_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");

            float[] gy_actual = y.GradState;

            AssertError.Tolerance(gy_expect, gy_actual, 1e-7f, 1e-5f, $"not equal gy");
        }

        float[] gx_expect = {
            -2.400000e+01f, -2.300000e+01f, -2.200000e+01f, -2.100000e+01f, -2.000000e+01f, -1.900000e+01f,
            -1.800000e+01f, -1.700000e+01f, -1.600000e+01f, -1.500000e+01f, -1.400000e+01f, -1.300000e+01f,
        };

        float[] gy_expect = {
            8.500000e+00f, 4.500000e+00f, 5.000000e-01f, -3.500000e+00f, -7.500000e+00f, -1.150000e+01f,
            -1.550000e+01f, -1.950000e+01f, -2.350000e+01f, -2.750000e+01f, -3.150000e+01f, -3.550000e+01f,
        };
    }
}
