using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class CosTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => ((float)idx - 12) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = Cos(x);
            Field err = y_expect - y_actual;

            StoreField n = err;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] div = n.State;

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            4.54648713e-01f,
            4.49801585e-01f,
            4.36022574e-01f,
            4.13542648e-01f,
            3.82907317e-01f,
            3.44970595e-01f,
            3.00879108e-01f,
            2.52046679e-01f,
            2.00120003e-01f,
            1.46936285e-01f,
            9.44739598e-02f,
            4.47978131e-02f,
            -0.00000000e+00f,
            -3.78614034e-02f,
            -6.68246043e-02f,
            -8.50852948e-02f,
            -9.10551037e-02f,
            -8.34156107e-02f,
            -6.11663385e-02f,
            -2.36652887e-02f,
            2.93392185e-02f,
            9.76864217e-02f,
            1.80791470e-01f,
            2.77644735e-01f,
        };
    }
}
