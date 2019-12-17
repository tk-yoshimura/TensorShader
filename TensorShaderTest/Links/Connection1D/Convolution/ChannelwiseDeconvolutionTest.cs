using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection1D {
    [TestClass]
    public class ChannelwiseDeconvolutionTest {
        [TestMethod]
        public void ReferenceTest() {
            int channels = 7, kwidth = 3, stride = 2, inwidth = 13;
            int outwidth = inwidth - kwidth + 1, batch = 2;

            float[] xval = (new float[inwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(channels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(channels, outwidth, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel1D(channels, 1, kwidth), wval);

            VariableField x_actual = xtensor;
            ParameterField w = wtensor;
            ParameterField y = ytensor;

            Field x_expect = ChannelwiseDeconvolution1D(y, w, stride, Shape.Map1D(channels, inwidth, batch));
            Field err = Abs(x_expect - x_actual);
            OutputNode err_node = err.Value.Save();

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] err_actual = err_node.Tensor.State;
            float[] gy_actual = y.GradTensor.State;
            float[] gw_actual = w.GradTensor.State;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gy_expect, gy_actual, 1e-7f, 1e-5f, $"not equal gy");
        }

        float[] gy_expect = new float[] {
            -1.7416000e-04f,  -1.8871000e-04f,  -1.9743000e-04f,  -2.0029600e-04f,  -1.9728400e-04f,  -1.8837000e-04f,  -1.7353000e-04f,
            -7.1508500e-04f,  -6.8824000e-04f,  -6.5555500e-04f,  -6.1700000e-04f,  -5.7254500e-04f,  -5.2216000e-04f,  -4.6581500e-04f,
            -1.2551700e-03f,  -1.1872000e-03f,  -1.1133200e-03f,  -1.0335000e-03f,  -9.4771000e-04f,  -8.5592000e-04f,  -7.5810000e-04f,
            -1.7952550e-03f,  -1.6861600e-03f,  -1.5710850e-03f,  -1.4500000e-03f,  -1.3228750e-03f,  -1.1896800e-03f,  -1.0503850e-03f,
            -2.3353400e-03f,  -2.1851200e-03f,  -2.0288500e-03f,  -1.8665000e-03f,  -1.6980400e-03f,  -1.5234400e-03f,  -1.3426700e-03f,
            -2.8804650e-03f,  -2.6881650e-03f,  -2.4897830e-03f,  -2.2852950e-03f,  -2.0746770e-03f,  -1.8579050e-03f,  -1.6349550e-03f,
            -3.6927100e-03f,  -3.4384600e-03f,  -3.1780440e-03f,  -2.9114380e-03f,  -2.6386180e-03f,  -2.3595600e-03f,  -2.0742400e-03f,
            -4.2285950e-03f,  -3.9340000e-03f,  -3.6331450e-03f,  -3.3260000e-03f,  -3.0125350e-03f,  -2.6927200e-03f,  -2.3665250e-03f,
            -4.7686800e-03f,  -4.4329600e-03f,  -4.0909100e-03f,  -3.7425000e-03f,  -3.3877000e-03f,  -3.0264800e-03f,  -2.6588100e-03f,
            -5.3087650e-03f,  -4.9319200e-03f,  -4.5486750e-03f,  -4.1590000e-03f,  -3.7628650e-03f,  -3.3602400e-03f,  -2.9510950e-03f,
            -5.8488500e-03f,  -5.4308800e-03f,  -5.0064400e-03f,  -4.5755000e-03f,  -4.1380300e-03f,  -3.6940000e-03f,  -3.2433800e-03f,
            -6.3990150e-03f,  -5.9379150e-03f,  -5.4703970e-03f,  -4.9964370e-03f,  -4.5160110e-03f,  -4.0290950e-03f,  -3.5356650e-03f,
        };

        float[] gw_expect = new float[] {
            -5.1470580e-02f,  -5.2933780e-02f,  -5.4423900e-02f,  -5.5941072e-02f,  -5.7485428e-02f,  -5.9057100e-02f,  -6.0656220e-02f,
            -5.4998678e-02f,  -5.6536240e-02f,  -5.8099410e-02f,  -5.9688260e-02f,  -6.1302862e-02f,  -6.2943288e-02f,  -6.4609610e-02f,
            -5.8004436e-02f,  -5.9637820e-02f,  -6.1298180e-02f,  -6.2985648e-02f,  -6.4700356e-02f,  -6.6442436e-02f,  -6.8212020e-02f,
        };
    }
}
