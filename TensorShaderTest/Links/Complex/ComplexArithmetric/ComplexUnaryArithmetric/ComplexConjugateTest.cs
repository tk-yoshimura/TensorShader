using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ComplexArithmetric {
    [TestClass]
    public class ComplexConjugateTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 2).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = ComplexConjugate(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

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

            Field y_expect = ComplexCast(ComplexReal(x), -ComplexImag(x));
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -2.400000e+01f, -2.150000e+01f, -2.100000e+01f, -1.650000e+01f, -1.800000e+01f, -1.150000e+01f,
            -1.500000e+01f, -6.500000e+00f, -1.200000e+01f, -1.500000e+00f, -9.000000e+00f, 3.500000e+00f,
            -6.000000e+00f, 8.500000e+00f, -3.000000e+00f, 1.350000e+01f, 0.000000e+00f, 1.850000e+01f,
            3.000000e+00f, 2.350000e+01f, 6.000000e+00f, 2.850000e+01f, 9.000000e+00f, 3.350000e+01f,
        };
    }
}
