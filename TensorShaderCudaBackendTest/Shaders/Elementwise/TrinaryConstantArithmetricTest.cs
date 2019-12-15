using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderCudaBackend;
using TensorShaderCudaBackend.Shaders.Elementwise;

namespace TensorShaderCudaBackendTest.ShadersTest.ElementwiseTest {
    [TestClass]
    public class TrinaryConstantArithmetricTest {

        [TestMethod]
        public void ExecuteTest() {
            const int length = 1024;

            Random random = new Random(1234);

            float[] h_x = (new float[length]).Select((_) => (float)random.NextDouble()).ToArray();
            float[] h_y = new float[length];

            Shader shader = new TrinaryBiConstantArithmetric("muladd", "#y = c1 + c2 * #x;");

            CudaArray<float> d_x = h_x;
            CudaArray<float> d_y = new CudaArray<float>(length);

            shader.Execute(stream: null, 3.0f, 2.0f, d_x, d_y, (uint)length);

            d_y.Read(h_y);

            for (int i = 0; i < length; i++) {
                Assert.AreEqual(3 + 2 * h_x[i], h_y[i], 1e-5f);
            }

            Console.WriteLine(shader);
        }
    }
}
