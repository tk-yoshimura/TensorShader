using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShaderCudaBackend;
using TensorShaderCudaBackend.Shaders.Elementwise;

namespace TensorShaderCudaBackendTest.ShadersTest.ElementwiseTest {
    [TestClass]
    public class BinaryFactorArithmetricTest {

        [TestMethod]
        public void ExecuteTest() {
            const int length = 1024;

            Random random = new(1234);

            float[] h_c = new float[] { 2 };
            float[] h_x = (new float[length]).Select((_) => (float)random.NextDouble()).ToArray();
            float[] h_y = new float[length];

            Shader shader = new BinaryFactorArithmetric("addc", "#y = #x + c;");

            CudaArray<float> d_c = h_c;
            CudaArray<float> d_x = h_x;
            CudaArray<float> d_y = new(length);

            shader.Execute(stream: null, d_c, d_x, d_y, (uint)length);

            d_y.Read(h_y);

            for (int i = 0; i < length; i++) {
                Assert.AreEqual(h_x[i] + 2, h_y[i], 1e-5f);
            }

            Console.WriteLine(shader);
        }
    }
}
