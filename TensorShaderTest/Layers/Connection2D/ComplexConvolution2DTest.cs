using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Layers;

namespace TensorShaderTest.Layers {
    [TestClass]
    public class ComplexConvolution2DTest {
        [TestMethod]
        public void ExecuteTest() {
            int inchannels = 4, outchannels = 6, inwidth = 13, inheight = 17, kwidth = 3, kheight = 5, batch = 7;

            VariableField x = (Shape.Map2D(inchannels, inwidth, inheight, batch));

            Layer layer = new ComplexConvolution2D(inchannels, outchannels, kwidth, kheight, use_bias: true, pad_mode: PaddingMode.Edge, "conv");

            Field y = layer.Forward(x);

            (Flow flow, Parameters parameters) = Flow.Optimize(y);
            flow.Execute();

            Assert.AreEqual(2, parameters.Count);
            Assert.AreEqual(inchannels, layer.InChannels);
            Assert.AreEqual(outchannels, layer.OutChannels);
            Assert.AreEqual(kwidth, layer.Width);
            Assert.AreEqual(kheight, layer.Height);
        }
    }
}
