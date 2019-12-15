using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Layers;

namespace TensorShaderTest.Layers {
    [TestClass]
    public class ChannelwiseDeconvolution1DTest {
        [TestMethod]
        public void ExecuteTest() {
            int channels = 4, inwidth = 13, kwidth = 3, stride = 2, batch = 7;

            VariableField x = new Tensor(Shape.Map1D(channels, inwidth, batch));

            Layer layer = new ChannelwiseDeconvolution1D(channels, kwidth, stride, use_bias: true, pad_mode: PaddingMode.Edge, label: "conv");

            Field y = layer.Forward(x);

            (Flow flow, Parameters parameters) = Flow.Optimize(y);
            flow.Execute();

            Assert.AreEqual(2, parameters.Count);
            Assert.AreEqual(channels, layer.InChannels);
            Assert.AreEqual(channels, layer.OutChannels);
            Assert.AreEqual(kwidth, layer.Width);
        }
    }
}
