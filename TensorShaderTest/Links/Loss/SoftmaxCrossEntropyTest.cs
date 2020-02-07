using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Loss {
    [TestClass]
    public class SoftmaxCrossEntropyTest {
        [TestMethod]
        public void ReferenceMethod() {
            int channels = 3, batch = 4;

            float[] xval =
                { 1, 2, 3,
                  -2, 1, 3,
                  3, 2, -1,
                  1, 2, 2 };

            float[] tval =
                { 0, 0, 1,
                  0, 1, 0,
                  0, 1, 0,
                  1, 0, 0 };

            Tensor xtensor = new Tensor(Shape.Map0D(channels, batch), xval);
            Tensor ttensor = new Tensor(Shape.Map0D(channels, batch), tval);

            ParameterField x = xtensor;
            VariableField t = ttensor;

            StoreField loss = Sum(SoftmaxCrossEntropy(x, t), axes: new int[] { Axis.Map0D.Channels });
            (Flow flow, _) = Flow.Optimize(loss);

            flow.Execute();

            float[] loss_actual = loss.State;

            AssertError.Tolerance(loss_expect, loss_actual, 1e-6f, 1e-4f, $"not equal loss");

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-6f, 1e-4f, $"not equal gx");
        }

        float[] loss_expect = {
            4.07605964e-01f,
            2.13284523e+00f,
            1.32656264e+00f,
            1.86199480e+00f,
        };

        float[] gx_expect = {
            3.66969986e-02f,
            9.97527845e-02f,
            -1.36449783e-01f,
            1.25832545e-02f,
            -1.88010381e+00f,
            1.86752056e+00f,
            9.56981207e-01f,
            -9.74508930e-01f,
            1.75277222e-02f,
            -1.57271082e+00f,
            7.86355408e-01f,
            7.86355408e-01f,
        };
    }
}
