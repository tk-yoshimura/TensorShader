using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.QuaternionArithmetric {
    [TestClass]
    public class QuaternionRReluTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 2).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = QuaternionRRelu(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 2).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field r = QuaternionR(x), i = QuaternionI(x), j = QuaternionJ(x), k = QuaternionK(x);

            Field y_expect = QuaternionCast(Relu(r), i, j, k);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            0.000000000e+00f, -2.250000000e+01f, -2.100000000e+01f, -1.950000000e+01f, 0.000000000e+00f, -1.650000000e+01f,
            -1.500000000e+01f, -1.350000000e+01f, 0.000000000e+00f, -1.050000000e+01f, -9.000000000e+00f, -7.500000000e+00f,
            -6.000000000e+00f, -4.500000000e+00f, -3.000000000e+00f, -1.500000000e+00f, 0.000000000e+00f, 1.500000000e+00f,
            3.000000000e+00f, 4.500000000e+00f, 6.000000000e+00f, 7.500000000e+00f, 9.000000000e+00f, 1.050000000e+01f,
        };
    }
}
