using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Layers;

namespace TensorShaderTest.Layers {
    [TestClass]
    public class TrivectorDenseTest {
        [TestMethod]
        public void ExecuteTest() {
            int inchannels = 9, outchannels = 12, batch = 7;

            VariableField x = (Shape.Map0D(inchannels, batch));

            Layer layer = new TrivectorDense(inchannels, outchannels, use_bias: true, "fc");

            Field y = layer.Forward(x);

            (Flow flow, Parameters parameters) = Flow.Optimize(y);
            flow.Execute();

            Assert.AreEqual(2, parameters.Count);
            Assert.AreEqual(inchannels, layer.InChannels);
            Assert.AreEqual(outchannels, layer.OutChannels);
        }
    }
}
