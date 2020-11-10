using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Loss {
    [TestClass]
    public class HingeLossTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => idx * idx / 144f - 0.5f).ToArray();
            float[] tval = (new float[length]).Select((_, idx) => ((float)idx - 12) / 12).ToArray();

            ParameterField x = xval;
            VariableField t = tval;

            StoreField loss = HingeLoss(x, t);

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);

            flow.Execute();

            float[] loss_actual = loss.State;

            AssertError.Tolerance(loss_expect, loss_actual, 1e-7f, 1e-5f, $"not equal loss");

            float[] gx_actual = x.GradState;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] loss_expect = {
            5.00000000e-01f,
            5.48032407e-01f,
            6.06481481e-01f,
            6.71875000e-01f,
            7.40740741e-01f,
            8.09606481e-01f,
            8.75000000e-01f,
            9.33449074e-01f,
            9.81481481e-01f,
            1.01562500e+00f,
            1.03240741e+00f,
            1.02835648e+00f,
            1.00000000e+00f,
            9.43865741e-01f,
            8.56481481e-01f,
            7.34375000e-01f,
            5.74074074e-01f,
            3.72106481e-01f,
            1.25000000e-01f,
            0.00000000e+00f,
            0.00000000e+00f,
            0.00000000e+00f,
            0.00000000e+00f,
            0.00000000e+00f,
        };

        float[] gx_expect = {
            5.00000000e-01f,
            5.02363040e-01f,
            5.05401235e-01f,
            5.03906250e-01f,
            4.93827160e-01f,
            4.72270448e-01f,
            4.37500000e-01f,
            3.88937114e-01f,
            3.27160494e-01f,
            2.53906250e-01f,
            1.72067901e-01f,
            8.56963735e-02f,
            -0.00000000e+00f,
            -7.86554784e-02f,
            -1.42746914e-01f,
            -1.83593750e-01f,
            -1.91358025e-01f,
            -1.55044367e-01f,
            -6.25000000e-02f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
            -0.00000000e+00f,
        };
    }
}
