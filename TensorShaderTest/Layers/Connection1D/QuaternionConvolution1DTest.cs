using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Layers;

namespace TensorShaderTest.Layers {
    [TestClass]
    public class QuaternionConvolution1DTest {
        [TestMethod]
        public void ExecuteTest() {
            int inchannels = 8, outchannels = 12, inwidth = 13, kwidth = 3, stride = 2, batch = 7;

            VariableField x = new Tensor(Shape.Map1D(inchannels, inwidth, batch));

            Layer layer = new QuaternionConvolution1D(inchannels, outchannels, kwidth, stride, use_bias: true, pad_mode: PaddingMode.Edge, "conv");

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
