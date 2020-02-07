using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Quaternion {
    [TestClass]
    public class QuaternionCastTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length / 4]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length / 4]).Select((_, idx) => (float)idx * 3 - length).Reverse().ToArray();
            float[] zval = (new float[length / 4]).Select((_, idx) => (float)idx * 4 - length).ToArray();
            float[] wval = (new float[length / 4]).Select((_, idx) => (float)idx * 5 - length).Reverse().ToArray();

            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length / 4), xval);
            Tensor ytensor = new Tensor(Shape.Vector(length / 4), yval);
            Tensor ztensor = new Tensor(Shape.Vector(length / 4), zval);
            Tensor wtensor = new Tensor(Shape.Vector(length / 4), wval);

            Tensor ttensor = new Tensor(Shape.Vector(length), tval);

            ParameterField x = xtensor;
            ParameterField y = ytensor;
            ParameterField z = ztensor;
            ParameterField w = wtensor;
            VariableField t_actual = ttensor;

            Field t_expect = QuaternionCast(x, y, z, w);
            StoreField err = t_expect - t_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");

            float[] gy_actual = y.GradState;

            AssertError.Tolerance(gy_expect, gy_actual, 1e-7f, 1e-5f, $"not equal gy");

            float[] gz_actual = z.GradState;

            AssertError.Tolerance(gz_expect, gz_actual, 1e-7f, 1e-5f, $"not equal gz");

            float[] gw_actual = w.GradState;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");
        }

        float[] gx_expect = {
            -2.400000e+01f, -2.400000e+01f, -2.400000e+01f, -2.400000e+01f, -2.400000e+01f, -2.400000e+01f,
        };

        float[] gy_expect = {
            -9.500000e+00f, -1.450000e+01f, -1.950000e+01f, -2.450000e+01f, -2.950000e+01f, -3.450000e+01f,
        };

        float[] gz_expect = {
            -2.500000e+01f, -2.300000e+01f, -2.100000e+01f, -1.900000e+01f, -1.700000e+01f, -1.500000e+01f,
        };

        float[] gw_expect = {
            -5.000000e-01f, -7.500000e+00f, -1.450000e+01f, -2.150000e+01f, -2.850000e+01f, -3.550000e+01f,
        };
    }
}
