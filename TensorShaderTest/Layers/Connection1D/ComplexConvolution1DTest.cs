using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Layers;

namespace TensorShaderTest.Layers {
    [TestClass]
    public class ComplexConvolution1DTest {
        [TestMethod]
        public void ExecuteTest() {
            int inchannels = 4, outchannels = 6, inwidth = 13, kwidth = 3, batch = 7;

            VariableField x = Shape.Map1D(inchannels, inwidth, batch);

            Layer layer = new ComplexConvolution1D(inchannels, outchannels, kwidth, use_bias: true, pad_mode: PaddingMode.Edge, "conv");

            Field y = layer.Forward(x);

            (Flow flow, Parameters parameters) = Flow.Optimize(y);
            flow.Execute();

            Assert.AreEqual(2, parameters.Count);
            Assert.AreEqual(inchannels, layer.InChannels);
            Assert.AreEqual(outchannels, layer.OutChannels);
            Assert.AreEqual(kwidth, layer.Width);
        }
    }
}
