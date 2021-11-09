using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShaderCudaBackend;
using TensorShaderCudaBackend.Cudnn;

namespace TensorShaderCudaBackendTest.APITest {
    [TestClass()]
    public class CudnnControllerTests {
        [TestMethod()]
        public void ControllerTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            CudnnController controller = new(new Stream());
            controller.Dispose();
        }
    }
}