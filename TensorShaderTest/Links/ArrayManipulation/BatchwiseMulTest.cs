using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ArrayManipulation {
    [TestClass]
    public class BatchwiseMulTest {
        [TestMethod]
        public void ReferenceTest() {
            const int length = 12, ch = 4, batch = 5;

            float[] xval = (new float[length * ch * batch]).Select((_, idx) => (float)idx).ToArray();
            float[] vval = new float[batch] { 0, 0.2f, 0.6f, 0.4f, 1 };
            float[] yval = (new float[length * ch * batch]).Select((_, idx) => (float)idx).Reverse().ToArray();

            Tensor xtensor = new Tensor(new Shape(ShapeType.Map, length, ch, batch), xval);
            Tensor vtensor = new Tensor(new Shape(ShapeType.Vector, batch), vval);
            Tensor ytensor = new Tensor(new Shape(ShapeType.Map, length, ch, batch), yval);

            ParameterField x = xtensor;
            ParameterField v = vtensor;
            VariableField y = ytensor;

            Field f = BatchwiseMul(x, v);
            Field err = Abs(f - y);

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");

            float[] gv_actual = v.GradTensor.State;

            AssertError.Tolerance(gv_expect, gv_actual, 1e-6f, 1e-4f, $"not equal gv");
        }

        float[] gx_expect = {
            -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f,
            -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f,
            -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f,
            -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f,
            -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f,
            -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f,
            -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f,
            -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f,
            -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f,
            -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, -3.6280e+01f, -3.6040e+01f,
            -3.5800e+01f, -3.5560e+01f, -3.5320e+01f, -3.5080e+01f, -3.4840e+01f,
            -3.4600e+01f, -3.4360e+01f, -3.4120e+01f, -3.3880e+01f, -3.3640e+01f,
            -3.3400e+01f, -3.3160e+01f, -3.2920e+01f, -3.2680e+01f, -3.2440e+01f,
            -3.2200e+01f, -3.1960e+01f, -3.1720e+01f, -3.1480e+01f, -3.1240e+01f,
            -3.1000e+01f, -3.0760e+01f, -3.0520e+01f, -3.0280e+01f, -3.0040e+01f,
            -2.9800e+01f, -2.9560e+01f, -2.9320e+01f, -2.9080e+01f, -2.8840e+01f,
            -2.8600e+01f, -2.8360e+01f, -2.8120e+01f, -2.7880e+01f, -2.7640e+01f,
            -2.7400e+01f, -2.7160e+01f, -2.6920e+01f, -2.6680e+01f, -2.6440e+01f,
            -2.6200e+01f, -2.5960e+01f, -2.5720e+01f, -2.5480e+01f, -2.5240e+01f,
            -2.5000e+01f, -5.1240e+01f, -5.0280e+01f, -4.9320e+01f, -4.8360e+01f,
            -4.7400e+01f, -4.6440e+01f, -4.5480e+01f, -4.4520e+01f, -4.3560e+01f,
            -4.2600e+01f, -4.1640e+01f, -4.0680e+01f, -3.9720e+01f, -3.8760e+01f,
            -3.7800e+01f, -3.6840e+01f, -3.5880e+01f, -3.4920e+01f, -3.3960e+01f,
            -3.3000e+01f, -3.2040e+01f, -3.1080e+01f, -3.0120e+01f, -2.9160e+01f,
            -2.8200e+01f, -2.7240e+01f, -2.6280e+01f, -2.5320e+01f, -2.4360e+01f,
            -2.3400e+01f, -2.2440e+01f, -2.1480e+01f, -2.0520e+01f, -1.9560e+01f,
            -1.8600e+01f, -1.7640e+01f, -1.6680e+01f, -1.5720e+01f, -1.4760e+01f,
            -1.3800e+01f, -1.2840e+01f, -1.1880e+01f, -1.0920e+01f, -9.9600e+00f,
            -9.0000e+00f, -8.0400e+00f, -7.0800e+00f, -6.1200e+00f, -1.4960e+01f,
            -1.4400e+01f, -1.3840e+01f, -1.3280e+01f, -1.2720e+01f, -1.2160e+01f,
            -1.1600e+01f, -1.1040e+01f, -1.0480e+01f, -9.9200e+00f, -9.3600e+00f,
            -8.8000e+00f, -8.2400e+00f, -7.6800e+00f, -7.1200e+00f, -6.5600e+00f,
            -6.0000e+00f, -5.4400e+00f, -4.8800e+00f, -4.3200e+00f, -3.7600e+00f,
            -3.2000e+00f, -2.6400e+00f, -2.0800e+00f, -1.5200e+00f, -9.6000e-01f,
            -4.0000e-01f, 1.6000e-01f, 7.2000e-01f, 1.2800e+00f, 1.8400e+00f,
            2.4000e+00f, 2.9600e+00f, 3.5200e+00f, 4.0800e+00f, 4.6400e+00f,
            5.2000e+00f, 5.7600e+00f, 6.3200e+00f, 6.8800e+00f, 7.4400e+00f,
            8.0000e+00f, 8.5600e+00f, 9.1200e+00f, 9.6800e+00f, 1.0240e+01f,
            1.0800e+01f, 1.1360e+01f, 1.4500e+02f, 1.4700e+02f, 1.4900e+02f,
            1.5100e+02f, 1.5300e+02f, 1.5500e+02f, 1.5700e+02f, 1.5900e+02f,
            1.6100e+02f, 1.6300e+02f, 1.6500e+02f, 1.6700e+02f, 1.6900e+02f,
            1.7100e+02f, 1.7300e+02f, 1.7500e+02f, 1.7700e+02f, 1.7900e+02f,
            1.8100e+02f, 1.8300e+02f, 1.8500e+02f, 1.8700e+02f, 1.8900e+02f,
            1.9100e+02f, 1.9300e+02f, 1.9500e+02f, 1.9700e+02f, 1.9900e+02f,
            2.0100e+02f, 2.0300e+02f, 2.0500e+02f, 2.0700e+02f, 2.0900e+02f,
            2.1100e+02f, 2.1300e+02f, 2.1500e+02f, 2.1700e+02f, 2.1900e+02f,
            2.2100e+02f, 2.2300e+02f, 2.2500e+02f, 2.2700e+02f, 2.2900e+02f,
            2.3100e+02f, 2.3300e+02f, 2.3500e+02f, 2.3700e+02f, 2.3900e+02f,
        };

        float[] gv_expect = {
            -2.3387e+05f, -5.1473e+05f, -2.5944e+05f, -2.3283e+04f, 2.0045e+06f,
        };
    }
}
