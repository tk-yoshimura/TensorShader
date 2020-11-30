using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Channelwise {
    [TestClass]
    public class SoftmaxTest {
        [TestMethod]
        public void ReferenceMethod() {
            int channels = 3, indexes = 4;

            float[] xval =
                { 1, 2, 3,
                  -2, 1, 3,
                  3, 2, -1,
                  1, 2, 2 };

            float[] tval =
                { 0, 0, 1,
                  0, 1, 0,
                  0, 1, 0,
                  0, 1, 1 };

            ParameterField x = (Shape.Map0D(channels, indexes), xval);
            VariableField t = (Shape.Map0D(channels, indexes), tval);

            StoreField y = Softmax(x);
            Field err = y - t;
            (Flow flow, _) = Flow.Optimize(err);

            flow.Execute();

            float[] y_actual = y.State.Value;

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"not equal y");

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        readonly float[] y_expect = {
            9.00305732e-02f,
            2.44728471e-01f,
            6.65240956e-01f,
            5.89975040e-03f,
            1.18499655e-01f,
            8.75600595e-01f,
            7.21399184e-01f,
            2.65387929e-01f,
            1.32128870e-02f,
            1.55362403e-01f,
            4.22318798e-01f,
            4.22318798e-01f,
        };
        readonly float[] gx_expect = {
            2.20330445e-02f,
            9.77510046e-02f,
            -1.19784049e-01f,
            -3.87232461e-03f,
            -1.82934324e-01f,
            1.86806648e-01f,
            2.85504546e-01f,
            -2.81376559e-01f,
            -4.12798732e-03f,
            9.61935936e-02f,
            -4.80967968e-02f,
            -4.80967968e-02f,
        };
    }
}
