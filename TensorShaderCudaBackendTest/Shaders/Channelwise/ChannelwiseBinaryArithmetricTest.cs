using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderCudaBackend;
using TensorShaderCudaBackend.API;
using TensorShaderCudaBackend.Shaders.Channelwise;

namespace TensorShaderCudaBackendTest.ShadersTest.ChannelwiseTest {
    [TestClass]
    public class ChannelwiseBinaryArithmetricTest {

        [TestMethod]
        public void ExecuteTest() {
            int const_length = (int)(Cuda.CurrectDeviceProperty.ConstMemoryBytes / sizeof(float));

            foreach (uint vec_length in new int[] { 1, 3, 4, 5, 31, 32, 33, const_length - 2, const_length - 1, const_length, const_length + 1, const_length + 2 }) {
                uint map_length = vec_length * 5;

                Random random = new Random(1234);

                float[] h_v = (new float[vec_length]).Select((_) => (float)random.NextDouble()).ToArray();
                float[] h_x = (new float[map_length]).Select((_) => (float)random.NextDouble()).ToArray();
                float[] h_y = new float[map_length];

                Shader shader = new ChannelwiseBinaryArithmetric("vecadd", "#y = #v + #x;", vec_length);

                CudaArray<float> d_v = new CudaArray<float>(h_v);
                CudaArray<float> d_x = new CudaArray<float>(h_x);
                CudaArray<float> d_y = new CudaArray<float>(map_length);

                shader.Execute(stream: null, d_v, d_x, d_y, map_length);

                d_y.Read(h_y);

                for (int i = 0; i < map_length; i++) {
                    Assert.AreEqual(h_v[i % vec_length] + h_x[i], h_y[i], 1e-5f);
                }

                Console.WriteLine(shader);
            }
        }
    }
}
