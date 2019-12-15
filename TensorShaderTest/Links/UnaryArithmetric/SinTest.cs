using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class SinTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => ((float)idx - 12) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = Sin(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -4.54648713e-01f,
            -5.08220193e-01f,
            -5.53738333e-01f,
            -5.90208602e-01f,
            -6.16950161e-01f,
            -6.33604002e-01f,
            -6.40131133e-01f,
            -6.36800988e-01f,
            -6.24170550e-01f,
            -6.03054927e-01f,
            -5.74490362e-01f,
            -5.39690885e-01f,
            -5.00000000e-01f,
            -4.56838901e-01f,
            -4.11652870e-01f,
            -3.65857494e-01f,
            -3.20786396e-01f,
            -2.77642079e-01f,
            -2.37451429e-01f,
            -2.01027258e-01f,
            -1.68937100e-01f,
            -1.41480267e-01f,
            -1.18673912e-01f,
            -1.00248598e-01f,
        };
    }
}
