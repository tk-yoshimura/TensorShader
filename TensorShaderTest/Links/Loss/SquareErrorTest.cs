using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Loss {
    [TestClass]
    public class SquareErrorTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)(idx - 12) * (idx - 12) / 8 + 1).ToArray();
            float[] tval = (new float[length]).Select((_, idx) => (float)idx * 2).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);
            Tensor ttensor = new Tensor(Shape.Vector(length), tval);

            ParameterField x = xtensor;
            VariableField t = ttensor;

            Field loss = SquareError(x, t);
            StoreField lossnode = loss;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);

            flow.Execute();

            float[] loss_actual = lossnode.State;

            AssertError.Tolerance(loss_expect, loss_actual, 1e-7f, 1e-5f, $"not equal loss");

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] loss_expect = {
            3.61000000e+02f,
            1.99515625e+02f,
            9.02500000e+01f,
            2.62656250e+01f,
            1.00000000e+00f,
            8.26562500e+00f,
            4.22500000e+01f,
            9.75156250e+01f,
            1.69000000e+02f,
            2.52015625e+02f,
            3.42250000e+02f,
            4.35765625e+02f,
            5.29000000e+02f,
            6.18765625e+02f,
            7.02250000e+02f,
            7.77015625e+02f,
            8.41000000e+02f,
            8.92515625e+02f,
            9.30250000e+02f,
            9.53265625e+02f,
            9.61000000e+02f,
            9.53265625e+02f,
            9.30250000e+02f,
            8.92515625e+02f,
        };

        float[] gx_expect = {
            1.37180000e+04f,
            5.63631641e+03f,
            1.71475000e+03f,
            2.69222656e+02f,
            2.00000000e+00f,
            -4.75273438e+01f,
            -5.49250000e+02f,
            -1.92593359e+03f,
            -4.39400000e+03f,
            -8.00149609e+03f,
            -1.26632500e+04f,
            -1.81932148e+04f,
            -2.43340000e+04f,
            -3.07835898e+04f,
            -3.72192500e+04f,
            -4.33186211e+04f,
            -4.87780000e+04f,
            -5.33278086e+04f,
            -5.67452500e+04f,
            -5.88641523e+04f,
            -5.95820000e+04f,
            -5.88641523e+04f,
            -5.67452500e+04f,
            -5.33278086e+04f,
        };
    }
}
