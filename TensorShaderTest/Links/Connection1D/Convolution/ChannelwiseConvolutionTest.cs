using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection1D {
    [TestClass]
    public class ChannelwiseConvolutionTest {
        [TestMethod]
        public void ReferenceTest() {
            int channels = 7, kwidth = 3, inwidth = 13;
            int outwidth = inwidth - kwidth + 1, batch = 2;

            float[] xval = (new float[inwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(channels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(channels, outwidth, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel1D(channels, 1, kwidth), wval);

            ParameterField x = xtensor;
            ParameterField w = wtensor;
            VariableField y_actual = ytensor;

            Field y_expect = ChannelwiseConvolution1D(x, w);
            Field err = Abs(y_expect - y_actual);

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;
            float[] gw_actual = w.GradTensor.State;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = new float[] {
            3.5000000e-06f,  -1.5390000e-05f,  -3.2418000e-05f,  -4.7566000e-05f,  -6.0816000e-05f,  -7.2150000e-05f,  -8.1550000e-05f, 
            -1.2876500e-04f,  -1.5332200e-04f,  -1.7407100e-04f,  -1.9097600e-04f,  -2.0400100e-04f,  -2.1311000e-04f,  -2.1826700e-04f, 
            -3.4970600e-04f,  -3.6656000e-04f,  -3.7757600e-04f,  -3.8270000e-04f,  -3.8187800e-04f,  -3.7505600e-04f,  -3.6218000e-04f, 
            -6.1205900e-04f,  -6.0948800e-04f,  -6.0095300e-04f,  -5.8640000e-04f,  -5.6577500e-04f,  -5.3902400e-04f,  -5.0609300e-04f, 
            -8.7441200e-04f,  -8.5241600e-04f,  -8.2433000e-04f,  -7.9010000e-04f,  -7.4967200e-04f,  -7.0299200e-04f,  -6.5000600e-04f, 
            -1.1367650e-03f,  -1.0953440e-03f,  -1.0477070e-03f,  -9.9380000e-04f,  -9.3356900e-04f,  -8.6696000e-04f,  -7.9391900e-04f, 
            -1.3991180e-03f,  -1.3382720e-03f,  -1.2710840e-03f,  -1.1975000e-03f,  -1.1174660e-03f,  -1.0309280e-03f,  -9.3783200e-04f, 
            -1.6614710e-03f,  -1.5812000e-03f,  -1.4944610e-03f,  -1.4012000e-03f,  -1.3013630e-03f,  -1.1948960e-03f,  -1.0817450e-03f, 
            -1.9238240e-03f,  -1.8241280e-03f,  -1.7178380e-03f,  -1.6049000e-03f,  -1.4852600e-03f,  -1.3588640e-03f,  -1.2256580e-03f, 
            -2.1861770e-03f,  -2.0670560e-03f,  -1.9412150e-03f,  -1.8086000e-03f,  -1.6691570e-03f,  -1.5228320e-03f,  -1.3695710e-03f, 
            -2.4485300e-03f,  -2.3099840e-03f,  -2.1645920e-03f,  -2.0123000e-03f,  -1.8530540e-03f,  -1.6868000e-03f,  -1.5134840e-03f, 
            -1.2344430e-03f,  -1.1271900e-03f,  -1.0152890e-03f,  -8.9870400e-04f,  -7.7739900e-04f,  -6.5133800e-04f,  -5.2048500e-04f, 
            -4.0257000e-04f,  -3.4145000e-04f,  -2.7796400e-04f,  -2.1209400e-04f,  -1.4382200e-04f,  -7.3130000e-05f,  0.0000000e+00f, 
            -1.4655200e-03f,  -1.4161460e-03f,  -1.3643640e-03f,  -1.3101560e-03f,  -1.2535040e-03f,  -1.1943900e-03f,  -1.1327960e-03f, 
            -2.5526480e-03f,  -2.4387660e-03f,  -2.3199840e-03f,  -2.1962660e-03f,  -2.0675760e-03f,  -1.9338780e-03f,  -1.7951360e-03f, 
            -3.2142950e-03f,  -3.0206240e-03f,  -2.8194770e-03f,  -2.6108000e-03f,  -2.3945390e-03f,  -2.1706400e-03f,  -1.9390490e-03f, 
            -3.4766480e-03f,  -3.2635520e-03f,  -3.0428540e-03f,  -2.8145000e-03f,  -2.5784360e-03f,  -2.3346080e-03f,  -2.0829620e-03f, 
            -3.7390010e-03f,  -3.5064800e-03f,  -3.2662310e-03f,  -3.0182000e-03f,  -2.7623330e-03f,  -2.4985760e-03f,  -2.2268750e-03f, 
            -4.0013540e-03f,  -3.7494080e-03f,  -3.4896080e-03f,  -3.2219000e-03f,  -2.9462300e-03f,  -2.6625440e-03f,  -2.3707880e-03f, 
            -4.2637070e-03f,  -3.9923360e-03f,  -3.7129850e-03f,  -3.4256000e-03f,  -3.1301270e-03f,  -2.8265120e-03f,  -2.5147010e-03f, 
            -4.5260600e-03f,  -4.2352640e-03f,  -3.9363620e-03f,  -3.6293000e-03f,  -3.3140240e-03f,  -2.9904800e-03f,  -2.6586140e-03f, 
            -4.7884130e-03f,  -4.4781920e-03f,  -4.1597390e-03f,  -3.8330000e-03f,  -3.4979210e-03f,  -3.1544480e-03f,  -2.8025270e-03f, 
            -5.0507660e-03f,  -4.7211200e-03f,  -4.3831160e-03f,  -4.0367000e-03f,  -3.6818180e-03f,  -3.3184160e-03f,  -2.9464400e-03f, 
            -5.3131190e-03f,  -4.9640480e-03f,  -4.6064930e-03f,  -4.2404000e-03f,  -3.8657150e-03f,  -3.4823840e-03f,  -3.0903530e-03f, 
            -2.6300120e-03f,  -2.3804980e-03f,  -2.1252440e-03f,  -1.8642140e-03f,  -1.5973720e-03f,  -1.3246820e-03f,  -1.0461080e-03f, 
            -8.4327600e-04f,  -7.1007000e-04f,  -5.7395200e-04f,  -4.3490400e-04f,  -2.9290800e-04f,  -1.4794600e-04f,  0.0000000e+00f, 
        };

        float[] gw_expect = new float[] {
            -1.7142033e-01f,  -1.7533250e-01f,  -1.7930953e-01f,  -1.8335181e-01f,  -1.8745973e-01f,  -1.9163371e-01f,  -1.9587412e-01f, 
            -1.8222889e-01f,  -1.8632995e-01f,  -1.9049678e-01f,  -1.9472979e-01f,  -1.9902937e-01f,  -2.0339592e-01f,  -2.0782983e-01f, 
            -1.9303746e-01f,  -1.9732739e-01f,  -2.0168403e-01f,  -2.0610777e-01f,  -2.1059900e-01f,  -2.1515813e-01f,  -2.1978554e-01f, 
        };
    }
}
