using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ComplexArithmetric {
    [TestClass]
    public class ComplexMulTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx * 3 - length).Reverse().ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).ToArray();

            ParameterField x = new Tensor(Shape.Vector(length), xval);
            ParameterField y = new Tensor(Shape.Vector(length), yval);
            VariableField t = new Tensor(Shape.Vector(length), tval);

            Field xy = ComplexMul(x, y);
            Field err = xy - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");

            float[] gy_actual = y.GradState;

            AssertError.Tolerance(gy_expect, gy_actual, 1e-7f, 1e-5f, $"not equal gy");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx * 3 - length).Reverse().ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx / 2).ToArray();

            ParameterField x = new Tensor(Shape.Vector(length), xval);
            ParameterField y = new Tensor(Shape.Vector(length), yval);
            VariableField t = new Tensor(Shape.Vector(length), tval);

            Field x_real = ComplexReal(x), x_imag = ComplexImag(x);
            Field y_real = ComplexReal(y), y_imag = ComplexImag(y);

            Field xy = ComplexCast(x_real * y_real - x_imag * y_imag, x_imag * y_real + x_real * y_imag);
            Field err = xy - t;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");

            float[] gy_actual = y.GradState;

            AssertError.Tolerance(gy_expect, gy_actual, 1e-7f, 1e-5f, $"not equal gy");
        }

        float[] gx_expect = {
            -9.095700e+04f, -8.338050e+04f, -5.643300e+04f, -5.072850e+04f, -3.196500e+04f, -2.786850e+04f,
            -1.582500e+04f, -1.307250e+04f, -6.285000e+03f, -4.612500e+03f, -1.617000e+03f, -7.605000e+02f,
            -9.300000e+01f, 2.115000e+02f, 1.500000e+01f, 3.150000e+01f, 4.350000e+02f, 4.275000e+02f,
            2.895000e+03f, 3.127500e+03f, 9.123000e+03f, 9.859500e+03f, 2.084700e+04f, 2.235150e+04f,
        };

        float[] gy_expect = {
            4.771100e+04f, 4.453200e+04f, 2.828300e+04f, 2.607600e+04f, 1.498300e+04f, 1.357200e+04f,
            6.659000e+03f, 5.868000e+03f, 2.159000e+03f, 1.812000e+03f, 3.310000e+02f, 2.520000e+02f,
            2.300000e+01f, 3.600000e+01f, 8.300000e+01f, 1.200000e+01f, -6.410000e+02f, -9.720000e+02f,
            -3.301000e+03f, -4.068000e+03f, -9.049000e+03f, -1.042800e+04f, -1.903700e+04f, -2.120400e+04f,
        };
    }
}
