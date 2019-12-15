using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderCudaBackend;

namespace TensorShaderCudaBackendTest {
    [TestClass]
    public class StreamTest {
        [TestMethod]
        public void CreateTest() {
            Stream stream = new Stream();

            Assert.IsTrue(stream.IsValid);

            stream.Dispose();

            Assert.IsFalse(stream.IsValid);
        }
    }
}
