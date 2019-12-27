using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Aggregation {
    [TestClass]
    public class MinTest {
        [TestMethod]
        public void ReferenceTest() {
            int channels = 7, width = 3, height = 5, batch = 2;

            float[] xval = (new float[channels * width * height * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            Tensor xtensor = new Tensor(Shape.Map2D(channels, width, height, batch), xval);

            float[] yval = (new float[width * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            Tensor ytensor = new Tensor(Shape.Map0D(width, batch), yval);

            VariableField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = Min(x, new int[] { Axis.Map2D.Channels, Axis.Map2D.Height }, keepdims: false);
            Field err = y_expect - y_actual;

            StoreField errnode = err.Value.Save();

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] err_actual = errnode.State;

            AssertError.Tolerance(err_expect, err_actual, 1e-7f, 1e-5f, $"not equal err");
        }

        float[] err_expect = {
            0.0000e+00f,
            6.0000e-03f,
            1.2000e-02f,
            1.0200e-01f,
            1.0800e-01f,
            1.1400e-01f,
        };
    }
}
