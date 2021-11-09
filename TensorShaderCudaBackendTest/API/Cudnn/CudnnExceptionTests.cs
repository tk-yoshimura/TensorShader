using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShaderCudaBackend;
using TensorShaderCudaBackend.Cudnn;

namespace TensorShaderCudaBackendTest.APITest {
    [TestClass()]
    public class CudnnExceptionTests {
        [TestMethod()]
        public void GetMessageTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            Assert.ThrowsException<CudaException>(() => {
                ConvolutionDescriptor convdesc = new(DataType.Float, (0, 0), (0, 0), (0, 0));
            });
        }
    }
}