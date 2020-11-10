using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Layers;

namespace TensorShaderTest.Layers {
    [TestClass]
    public class PointwiseConvolution2DTest {
        [TestMethod]
        public void ExecuteTest() {
            int inchannels = 4, outchannels = 6, inwidth = 13, inheight = 17, batch = 7;

            VariableField x = Shape.Map2D(inchannels, inwidth, inheight, batch);

            Layer layer = new PointwiseConvolution2D(inchannels, outchannels, use_bias: true, "conv");

            Field y = layer.Forward(x);

            (Flow flow, Parameters parameters) = Flow.Optimize(y);
            flow.Execute();

            Assert.AreEqual(2, parameters.Count);
            Assert.AreEqual(inchannels, layer.InChannels);
            Assert.AreEqual(outchannels, layer.OutChannels);
        }
    }
}
