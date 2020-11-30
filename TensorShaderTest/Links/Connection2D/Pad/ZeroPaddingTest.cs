using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection2D {
    [TestClass]
    public class ZeroPaddingTest {
        [TestMethod]
        public void ReferenceTest() {
            int pad_left = 2, pad_right = 1, pad_top = 3, pad_bottom = 5;
            int channels = 5, inwidth = 6, inheight = 7, batch = 2;
            int outwidth = inwidth + pad_left + pad_right;
            int outheight = inheight + pad_top + pad_bottom;

            float[] xval = (new float[channels * inwidth * inheight * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[channels * outwidth * outheight * batch]).Select((_, idx) => idx * 2e-3f).ToArray();

            ParameterField x = (Shape.Map2D(channels, inwidth, inheight, batch), xval);
            VariableField y_actual = (Shape.Map2D(channels, outwidth, outheight, batch), yval);

            Field y_expect = ZeroPadding2D(x, pad_left, pad_right, pad_top, pad_bottom);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        readonly float[] gx_expect = {
            -2.9000e-01f, -2.9100e-01f, -2.9200e-01f, -2.9300e-01f, -2.9400e-01f,
            -2.9500e-01f, -2.9600e-01f, -2.9700e-01f, -2.9800e-01f, -2.9900e-01f,
            -3.0000e-01f, -3.0100e-01f, -3.0200e-01f, -3.0300e-01f, -3.0400e-01f,
            -3.0500e-01f, -3.0600e-01f, -3.0700e-01f, -3.0800e-01f, -3.0900e-01f,
            -3.1000e-01f, -3.1100e-01f, -3.1200e-01f, -3.1300e-01f, -3.1400e-01f,
            -3.1500e-01f, -3.1600e-01f, -3.1700e-01f, -3.1800e-01f, -3.1900e-01f,
            -3.5000e-01f, -3.5100e-01f, -3.5200e-01f, -3.5300e-01f, -3.5400e-01f,
            -3.5500e-01f, -3.5600e-01f, -3.5700e-01f, -3.5800e-01f, -3.5900e-01f,
            -3.6000e-01f, -3.6100e-01f, -3.6200e-01f, -3.6300e-01f, -3.6400e-01f,
            -3.6500e-01f, -3.6600e-01f, -3.6700e-01f, -3.6800e-01f, -3.6900e-01f,
            -3.7000e-01f, -3.7100e-01f, -3.7200e-01f, -3.7300e-01f, -3.7400e-01f,
            -3.7500e-01f, -3.7600e-01f, -3.7700e-01f, -3.7800e-01f, -3.7900e-01f,
            -4.1000e-01f, -4.1100e-01f, -4.1200e-01f, -4.1300e-01f, -4.1400e-01f,
            -4.1500e-01f, -4.1600e-01f, -4.1700e-01f, -4.1800e-01f, -4.1900e-01f,
            -4.2000e-01f, -4.2100e-01f, -4.2200e-01f, -4.2300e-01f, -4.2400e-01f,
            -4.2500e-01f, -4.2600e-01f, -4.2700e-01f, -4.2800e-01f, -4.2900e-01f,
            -4.3000e-01f, -4.3100e-01f, -4.3200e-01f, -4.3300e-01f, -4.3400e-01f,
            -4.3500e-01f, -4.3600e-01f, -4.3700e-01f, -4.3800e-01f, -4.3900e-01f,
            -4.7000e-01f, -4.7100e-01f, -4.7200e-01f, -4.7300e-01f, -4.7400e-01f,
            -4.7500e-01f, -4.7600e-01f, -4.7700e-01f, -4.7800e-01f, -4.7900e-01f,
            -4.8000e-01f, -4.8100e-01f, -4.8200e-01f, -4.8300e-01f, -4.8400e-01f,
            -4.8500e-01f, -4.8600e-01f, -4.8700e-01f, -4.8800e-01f, -4.8900e-01f,
            -4.9000e-01f, -4.9100e-01f, -4.9200e-01f, -4.9300e-01f, -4.9400e-01f,
            -4.9500e-01f, -4.9600e-01f, -4.9700e-01f, -4.9800e-01f, -4.9900e-01f,
            -5.3000e-01f, -5.3100e-01f, -5.3200e-01f, -5.3300e-01f, -5.3400e-01f,
            -5.3500e-01f, -5.3600e-01f, -5.3700e-01f, -5.3800e-01f, -5.3900e-01f,
            -5.4000e-01f, -5.4100e-01f, -5.4200e-01f, -5.4300e-01f, -5.4400e-01f,
            -5.4500e-01f, -5.4600e-01f, -5.4700e-01f, -5.4800e-01f, -5.4900e-01f,
            -5.5000e-01f, -5.5100e-01f, -5.5200e-01f, -5.5300e-01f, -5.5400e-01f,
            -5.5500e-01f, -5.5600e-01f, -5.5700e-01f, -5.5800e-01f, -5.5900e-01f,
            -5.9000e-01f, -5.9100e-01f, -5.9200e-01f, -5.9300e-01f, -5.9400e-01f,
            -5.9500e-01f, -5.9600e-01f, -5.9700e-01f, -5.9800e-01f, -5.9900e-01f,
            -6.0000e-01f, -6.0100e-01f, -6.0200e-01f, -6.0300e-01f, -6.0400e-01f,
            -6.0500e-01f, -6.0600e-01f, -6.0700e-01f, -6.0800e-01f, -6.0900e-01f,
            -6.1000e-01f, -6.1100e-01f, -6.1200e-01f, -6.1300e-01f, -6.1400e-01f,
            -6.1500e-01f, -6.1600e-01f, -6.1700e-01f, -6.1800e-01f, -6.1900e-01f,
            -6.5000e-01f, -6.5100e-01f, -6.5200e-01f, -6.5300e-01f, -6.5400e-01f,
            -6.5500e-01f, -6.5600e-01f, -6.5700e-01f, -6.5800e-01f, -6.5900e-01f,
            -6.6000e-01f, -6.6100e-01f, -6.6200e-01f, -6.6300e-01f, -6.6400e-01f,
            -6.6500e-01f, -6.6600e-01f, -6.6700e-01f, -6.6800e-01f, -6.6900e-01f,
            -6.7000e-01f, -6.7100e-01f, -6.7200e-01f, -6.7300e-01f, -6.7400e-01f,
            -6.7500e-01f, -6.7600e-01f, -6.7700e-01f, -6.7800e-01f, -6.7900e-01f,
            -1.4300e+00f, -1.4310e+00f, -1.4320e+00f, -1.4330e+00f, -1.4340e+00f,
            -1.4350e+00f, -1.4360e+00f, -1.4370e+00f, -1.4380e+00f, -1.4390e+00f,
            -1.4400e+00f, -1.4410e+00f, -1.4420e+00f, -1.4430e+00f, -1.4440e+00f,
            -1.4450e+00f, -1.4460e+00f, -1.4470e+00f, -1.4480e+00f, -1.4490e+00f,
            -1.4500e+00f, -1.4510e+00f, -1.4520e+00f, -1.4530e+00f, -1.4540e+00f,
            -1.4550e+00f, -1.4560e+00f, -1.4570e+00f, -1.4580e+00f, -1.4590e+00f,
            -1.4900e+00f, -1.4910e+00f, -1.4920e+00f, -1.4930e+00f, -1.4940e+00f,
            -1.4950e+00f, -1.4960e+00f, -1.4970e+00f, -1.4980e+00f, -1.4990e+00f,
            -1.5000e+00f, -1.5010e+00f, -1.5020e+00f, -1.5030e+00f, -1.5040e+00f,
            -1.5050e+00f, -1.5060e+00f, -1.5070e+00f, -1.5080e+00f, -1.5090e+00f,
            -1.5100e+00f, -1.5110e+00f, -1.5120e+00f, -1.5130e+00f, -1.5140e+00f,
            -1.5150e+00f, -1.5160e+00f, -1.5170e+00f, -1.5180e+00f, -1.5190e+00f,
            -1.5500e+00f, -1.5510e+00f, -1.5520e+00f, -1.5530e+00f, -1.5540e+00f,
            -1.5550e+00f, -1.5560e+00f, -1.5570e+00f, -1.5580e+00f, -1.5590e+00f,
            -1.5600e+00f, -1.5610e+00f, -1.5620e+00f, -1.5630e+00f, -1.5640e+00f,
            -1.5650e+00f, -1.5660e+00f, -1.5670e+00f, -1.5680e+00f, -1.5690e+00f,
            -1.5700e+00f, -1.5710e+00f, -1.5720e+00f, -1.5730e+00f, -1.5740e+00f,
            -1.5750e+00f, -1.5760e+00f, -1.5770e+00f, -1.5780e+00f, -1.5790e+00f,
            -1.6100e+00f, -1.6110e+00f, -1.6120e+00f, -1.6130e+00f, -1.6140e+00f,
            -1.6150e+00f, -1.6160e+00f, -1.6170e+00f, -1.6180e+00f, -1.6190e+00f,
            -1.6200e+00f, -1.6210e+00f, -1.6220e+00f, -1.6230e+00f, -1.6240e+00f,
            -1.6250e+00f, -1.6260e+00f, -1.6270e+00f, -1.6280e+00f, -1.6290e+00f,
            -1.6300e+00f, -1.6310e+00f, -1.6320e+00f, -1.6330e+00f, -1.6340e+00f,
            -1.6350e+00f, -1.6360e+00f, -1.6370e+00f, -1.6380e+00f, -1.6390e+00f,
            -1.6700e+00f, -1.6710e+00f, -1.6720e+00f, -1.6730e+00f, -1.6740e+00f,
            -1.6750e+00f, -1.6760e+00f, -1.6770e+00f, -1.6780e+00f, -1.6790e+00f,
            -1.6800e+00f, -1.6810e+00f, -1.6820e+00f, -1.6830e+00f, -1.6840e+00f,
            -1.6850e+00f, -1.6860e+00f, -1.6870e+00f, -1.6880e+00f, -1.6890e+00f,
            -1.6900e+00f, -1.6910e+00f, -1.6920e+00f, -1.6930e+00f, -1.6940e+00f,
            -1.6950e+00f, -1.6960e+00f, -1.6970e+00f, -1.6980e+00f, -1.6990e+00f,
            -1.7300e+00f, -1.7310e+00f, -1.7320e+00f, -1.7330e+00f, -1.7340e+00f,
            -1.7350e+00f, -1.7360e+00f, -1.7370e+00f, -1.7380e+00f, -1.7390e+00f,
            -1.7400e+00f, -1.7410e+00f, -1.7420e+00f, -1.7430e+00f, -1.7440e+00f,
            -1.7450e+00f, -1.7460e+00f, -1.7470e+00f, -1.7480e+00f, -1.7490e+00f,
            -1.7500e+00f, -1.7510e+00f, -1.7520e+00f, -1.7530e+00f, -1.7540e+00f,
            -1.7550e+00f, -1.7560e+00f, -1.7570e+00f, -1.7580e+00f, -1.7590e+00f,
            -1.7900e+00f, -1.7910e+00f, -1.7920e+00f, -1.7930e+00f, -1.7940e+00f,
            -1.7950e+00f, -1.7960e+00f, -1.7970e+00f, -1.7980e+00f, -1.7990e+00f,
            -1.8000e+00f, -1.8010e+00f, -1.8020e+00f, -1.8030e+00f, -1.8040e+00f,
            -1.8050e+00f, -1.8060e+00f, -1.8070e+00f, -1.8080e+00f, -1.8090e+00f,
            -1.8100e+00f, -1.8110e+00f, -1.8120e+00f, -1.8130e+00f, -1.8140e+00f,
            -1.8150e+00f, -1.8160e+00f, -1.8170e+00f, -1.8180e+00f, -1.8190e+00f,
        };
    }
}
