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
            2.2750000e-06f,  -9.7200000e-06f,  -1.9811000e-05f,  -2.7980000e-05f,  -3.4209000e-05f,  -3.8480000e-05f,  -4.0775000e-05f,
            -1.2453000e-04f,  -1.4286400e-04f,  -1.5730600e-04f,  -1.6782000e-04f,  -1.7437000e-04f,  -1.7692000e-04f,  -1.7543400e-04f,
            -8.1627000e-05f,  -8.7672000e-05f,  -9.1729000e-05f,  -9.3780000e-05f,  -9.3807000e-05f,  -9.1792000e-05f,  -8.7717000e-05f,
            -2.9233400e-04f,  -2.9876800e-04f,  -3.0114200e-04f,  -2.9942000e-04f,  -2.9356600e-04f,  -2.8354400e-04f,  -2.6931800e-04f,
            -1.6552900e-04f,  -1.6562400e-04f,  -1.6364700e-04f,  -1.5958000e-04f,  -1.5340500e-04f,  -1.4510400e-04f,  -1.3465900e-04f,
            -4.6013800e-04f,  -4.5467200e-04f,  -4.4497800e-04f,  -4.3102000e-04f,  -4.1276200e-04f,  -3.9016800e-04f,  -3.6320200e-04f,
            -2.4943100e-04f,  -2.4357600e-04f,  -2.3556500e-04f,  -2.2538000e-04f,  -2.1300300e-04f,  -1.9841600e-04f,  -1.8160100e-04f,
            -6.2794200e-04f,  -6.1057600e-04f,  -5.8881400e-04f,  -5.6262000e-04f,  -5.3195800e-04f,  -4.9679200e-04f,  -4.5708600e-04f,
            -3.3333300e-04f,  -3.2152800e-04f,  -3.0748300e-04f,  -2.9118000e-04f,  -2.7260100e-04f,  -2.5172800e-04f,  -2.2854300e-04f,
            -7.9574600e-04f,  -7.6648000e-04f,  -7.3265000e-04f,  -6.9422000e-04f,  -6.5115400e-04f,  -6.0341600e-04f,  -5.5097000e-04f,
            -4.1723500e-04f,  -3.9948000e-04f,  -3.7940100e-04f,  -3.5698000e-04f,  -3.3219900e-04f,  -3.0504000e-04f,  -2.7548500e-04f,
            -1.9257000e-04f,  -1.6645000e-04f,  -1.3796400e-04f,  -1.0709400e-04f,  -7.3822000e-05f,  -3.8130000e-05f,  0.0000000e+00f,
            -7.6552000e-04f,  -7.5114600e-04f,  -7.3436400e-04f,  -7.1515600e-04f,  -6.9350400e-04f,  -6.6939000e-04f,  -6.4279600e-04f,
            -4.9758800e-04f,  -4.7440800e-04f,  -4.4877800e-04f,  -4.2068000e-04f,  -3.9009600e-04f,  -3.5700800e-04f,  -3.2139800e-04f,
            -1.1242560e-03f,  -1.0722400e-03f,  -1.0152400e-03f,  -9.5322000e-04f,  -8.8614400e-04f,  -8.1397600e-04f,  -7.3668000e-04f,
            -5.8149000e-04f,  -5.5236000e-04f,  -5.2069600e-04f,  -4.8648000e-04f,  -4.4969400e-04f,  -4.1032000e-04f,  -3.6834000e-04f,
            -1.2920600e-03f,  -1.2281440e-03f,  -1.1590760e-03f,  -1.0848200e-03f,  -1.0053400e-03f,  -9.2060000e-04f,  -8.3056400e-04f,
            -6.6539200e-04f,  -6.3031200e-04f,  -5.9261400e-04f,  -5.5228000e-04f,  -5.0929200e-04f,  -4.6363200e-04f,  -4.1528200e-04f,
            -1.4598640e-03f,  -1.3840480e-03f,  -1.3029120e-03f,  -1.2164200e-03f,  -1.1245360e-03f,  -1.0272240e-03f,  -9.2444800e-04f,
            -7.4929400e-04f,  -7.0826400e-04f,  -6.6453200e-04f,  -6.1808000e-04f,  -5.6889000e-04f,  -5.1694400e-04f,  -4.6222400e-04f,
            -1.6276680e-03f,  -1.5399520e-03f,  -1.4467480e-03f,  -1.3480200e-03f,  -1.2437320e-03f,  -1.1338480e-03f,  -1.0183320e-03f,
            -8.3319600e-04f,  -7.8621600e-04f,  -7.3645000e-04f,  -6.8388000e-04f,  -6.2848800e-04f,  -5.7025600e-04f,  -5.0916600e-04f,
            -1.7954720e-03f,  -1.6958560e-03f,  -1.5905840e-03f,  -1.4796200e-03f,  -1.3629280e-03f,  -1.2404720e-03f,  -1.1122160e-03f,
            -9.1709800e-04f,  -8.6416800e-04f,  -8.0836800e-04f,  -7.4968000e-04f,  -6.8808600e-04f,  -6.2356800e-04f,  -5.5610800e-04f,
            -4.2327600e-04f,  -3.6007000e-04f,  -2.9395200e-04f,  -2.2490400e-04f,  -1.5290800e-04f,  -7.7946000e-05f,  0.0000000e+00f,
        };

        float[] gw_expect = new float[] {
            -4.7648776e-02f,  -4.9365628e-02f,  -5.1117856e-02f,  -5.2905676e-02f,  -5.4729304e-02f,  -5.6588956e-02f,  -5.8484848e-02f,
            -5.0604358e-02f,  -5.2424236e-02f,  -5.4279994e-02f,  -5.6171848e-02f,  -5.8100014e-02f,  -6.0064708e-02f,  -6.2066146e-02f,
            -5.3559940e-02f,  -5.5482844e-02f,  -5.7442132e-02f,  -5.9438020e-02f,  -6.1470724e-02f,  -6.3540460e-02f,  -6.5647444e-02f,
        };
    }
}
