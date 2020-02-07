using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class Log10Test {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => idx + 0.5f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = Log10(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -2.61471332e-01f,
            3.89199058e-02f,
            5.46527772e-02f,
            5.19996969e-02f,
            4.69564837e-02f,
            4.20104388e-02f,
            3.76107946e-02f,
            3.37820072e-02f,
            3.04560804e-02f,
            2.75536354e-02f,
            2.50038899e-02f,
            2.27481375e-02f,
            2.07387780e-02f,
            1.89373981e-02f,
            1.73129150e-02f,
            1.58400282e-02f,
            1.44979942e-02f,
            1.32696747e-02f,
            1.21408015e-02f,
            1.10994076e-02f,
            1.01353851e-02f,
            9.24013949e-03f,
            8.40631694e-03f,
            7.62758844e-03f,
        };
    }
}
