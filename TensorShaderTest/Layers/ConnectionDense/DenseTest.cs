using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Layers;

namespace TensorShaderTest.Layers {
    [TestClass]
    public class DenseTest {
        [TestMethod]
        public void ExecuteTest() {
            int inchannels = 4, outchannels = 6, batch = 7;

            VariableField x = Shape.Map0D(inchannels, batch);

            Layer layer = new Dense(inchannels, outchannels, use_bias: true, "fc");

            Field y = layer.Forward(x);

            (Flow flow, Parameters parameters) = Flow.Optimize(y);
            flow.Execute();

            Assert.AreEqual(2, parameters.Count);
            Assert.AreEqual(inchannels, layer.InChannels);
            Assert.AreEqual(outchannels, layer.OutChannels);
        }
    }
}
