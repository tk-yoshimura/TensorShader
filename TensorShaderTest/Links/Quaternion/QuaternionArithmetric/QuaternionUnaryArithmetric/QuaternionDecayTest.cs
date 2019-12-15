using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.QuaternionArithmetric {
    [TestClass]
    public class QuaternionDecayTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);
            Tensor ttensor = new Tensor(Shape.Vector(length), tval);

            ParameterField x = xtensor;
            VariableField t = ttensor;

            Field xy = QuaternionDecay(x);
            Field err = xy - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);
            Tensor ttensor = new Tensor(Shape.Vector(length), tval);

            ParameterField x = xtensor;
            VariableField t = ttensor;

            Field xr = QuaternionR(x), xi = QuaternionI(x), xj = QuaternionJ(x), xk = QuaternionK(x);
            Field norm = Square(xr) + Square(xi) + Square(xj) + Square(xk);
            Field s = norm / (norm + 1);

            Field xy = QuaternionCast(
                  xr * s,
                  xi * s,
                  xj * s,
                  xk * s);
            Field err = xy - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -3.550721372e+01f, -3.300635580e+01f, -3.050549789e+01f, -2.800463998e+01f, -2.551657142e+01f, -2.301351362e+01f,
            -2.051045583e+01f, -1.800739803e+01f, -1.558938134e+01f, -1.305567237e+01f, -1.052196340e+01f, -7.988254426e+00f,
            -5.403508772e+00f, -2.978125523e+00f, -5.527422743e-01f, 1.872640975e+00f, 4.499558991e+00f, 6.996725966e+00f,
            9.493892942e+00f, 1.199105992e+01f, 1.450022622e+01f, 1.699978522e+01f, 1.949934421e+01f, 2.199890320e+01f,
        };
    }
}
