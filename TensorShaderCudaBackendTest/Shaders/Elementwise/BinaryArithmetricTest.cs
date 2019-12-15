using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderCudaBackend;
using TensorShaderCudaBackend.Shaders.Elementwise;

namespace TensorShaderCudaBackendTest.ShadersTest.ElementwiseTest {
    [TestClass]
    public class BinaryArithmetricTest {

        [TestMethod]
        public void ExecuteTest() {
            const int length = 1024;

            Random random = new Random(1234);

            float[] h_x1 = (new float[length]).Select((_) => (float)random.NextDouble()).ToArray();
            float[] h_x2 = (new float[length]).Select((_) => (float)random.NextDouble()).ToArray();
            float[] h_y = new float[length];

            Shader shader = new BinaryArithmetric("add1mul", "#y = #x1 * (#x2 + 1);");

            CudaArray<float> d_x1 = h_x1;
            CudaArray<float> d_x2 = h_x2;
            CudaArray<float> d_y = new CudaArray<float>(length);

            shader.Execute(stream: null, d_x1, d_x2, d_y, (uint)length);

            d_y.Read(h_y);

            for (int i = 0; i < length; i++) {
                Assert.AreEqual(h_x1[i] * (h_x2[i] + 1), h_y[i], 1e-5f);
            }

            Console.WriteLine(shader);
        }
    }
}
