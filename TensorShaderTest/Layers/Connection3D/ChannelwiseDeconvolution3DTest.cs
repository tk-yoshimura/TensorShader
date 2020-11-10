using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Layers;

namespace TensorShaderTest.Layers {
    [TestClass]
    public class ChannelwiseDeconvolution3DTest {
        [TestMethod]
        public void ExecuteTest() {
            int channels = 4, inwidth = 13, inheight = 17, indepth = 19, kwidth = 3, kheight = 5, kdepth = 7, batch = 7;

            VariableField x = Shape.Map3D(channels, inwidth, inheight, indepth, batch);

            Layer layer = new ChannelwiseDeconvolution3D(channels, kwidth, kheight, kdepth, use_bias: true, pad_mode: PaddingMode.Edge, label: "conv");

            Field y = layer.Forward(x);

            (Flow flow, Parameters parameters) = Flow.Optimize(y);
            flow.Execute();

            Assert.AreEqual(2, parameters.Count);
            Assert.AreEqual(channels, layer.InChannels);
            Assert.AreEqual(channels, layer.OutChannels);
            Assert.AreEqual(kwidth, layer.Width);
            Assert.AreEqual(kheight, layer.Height);
            Assert.AreEqual(kdepth, layer.Depth);
        }
    }
}
