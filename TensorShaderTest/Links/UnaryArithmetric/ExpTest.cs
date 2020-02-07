using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.UnaryArithmetric {
    [TestClass]
    public class ExpTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => ((float)idx - 12) / 12f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor = new Tensor(Shape.Vector(length), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = Exp(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            1.35335283e-01f,
            1.43219344e-01f,
            1.52659085e-01f,
            1.64084341e-01f,
            1.78027618e-01f,
            1.95145902e-01f,
            2.16246776e-01f,
            2.42319691e-01f,
            2.74573349e-01f,
            3.14480366e-01f,
            3.63830592e-01f,
            4.24794702e-01f,
            5.00000000e-01f,
            5.92620719e-01f,
            7.06485518e-01f,
            8.46205385e-01f,
            1.01732576e+00f,
            1.22650733e+00f,
            1.48174088e+00f,
            1.79260243e+00f,
            2.17055619e+00f,
            2.62931406e+00f,
            3.18526215e+00f,
            3.85796677e+00f,
        };
    }
}
