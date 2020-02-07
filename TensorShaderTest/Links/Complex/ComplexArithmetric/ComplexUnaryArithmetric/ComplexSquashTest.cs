using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ComplexArithmetric {
    [TestClass]
    public class ComplexSquashTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 2 + 1).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = ComplexSquash(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-4f, $"not equal gx");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 2 + 1).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field x_real = ComplexReal(x), x_imag = ComplexImag(x);
            Field norm = Sqrt(Square(x_real) + Square(x_imag)) + 1;

            Field y_expect = ComplexCast(x_real / norm, x_imag / norm);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-4f, $"not equal gx");
        }

        float[] gx_expect = {
            -3.934559129e-03f,  -1.974806004e-02f,  -4.519841637e-03f,  -2.735929767e-02f,  -5.364476973e-03f,  -4.119388841e-02f,
            -6.876614193e-03f,  -7.091106449e-02f,  -1.179564237e-02f,  -1.565740045e-01f,  -8.072326007e-02f,  -6.342796223e-01f,
            -2.166666667e+00f,  -5.925925926e-01f,  -2.347818275e-01f,  4.363281346e-02f,  -7.382436603e-02f,  2.541987235e-02f,
            -3.374903262e-02f,  1.635576379e-02f,  -1.840283289e-02f,  1.168902576e-02f,  -1.108399406e-02f,  8.958104125e-03f,
        };
    }
}
