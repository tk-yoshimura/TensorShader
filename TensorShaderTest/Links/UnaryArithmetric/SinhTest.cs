using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class SinhTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (idx + 0.5f) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = Sinh(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            4.17149087e-02f,
            8.43135468e-02f,
            1.29265709e-01f,
            1.78137031e-01f,
            2.32634975e-01f,
            2.94671265e-01f,
            3.66430641e-01f,
            4.50448046e-01f,
            5.49696583e-01f,
            6.67688929e-01f,
            8.08595243e-01f,
            9.77381096e-01f,
            1.17996953e+00f,
            1.42343205e+00f,
            1.71621409e+00f,
            2.06840172e+00f,
            2.49203723e+00f,
            3.00149279e+00f,
            3.61391309e+00f,
            4.34973969e+00f,
            5.23333220e+00f,
            6.29370414e+00f,
            7.56539461e+00f,
            9.08950066e+00f,
        };
    }
}
