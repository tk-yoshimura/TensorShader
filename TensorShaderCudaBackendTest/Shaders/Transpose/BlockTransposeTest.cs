using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShaderCudaBackend;
using TensorShaderCudaBackend.Shaders.Transpose;

namespace TensorShaderCudaBackendTest.ShadersTest.TransposeTest {
    [TestClass]
    public class BlockTransposeTest {

        [TestMethod]
        public void ExecuteTest() {
            (uint block, uint n, uint m)[] tests = new (uint block, uint n, uint m)[] {
                (1, 1, 1),
                (1, 1, 15),
                (1, 15, 1),
                (15, 1, 1),
                (1, 15, 15),
                (15, 15, 1),
                (15, 1, 15),
                (15, 15, 15),
                (2, 31, 31),
                (3, 32, 32),
                (4, 33, 255),
                (7, 255, 31),
                (8, 256, 256),
                (15, 512, 32),
                (16, 32, 512),
                (32, 513, 513)
            };

            foreach ((uint block, uint n, uint m) in tests) {
                uint length = block * n * m;

                float[] h_x = (new float[length + 1]).Select((_, i) => (float)i).ToArray();
                float[] h_y = new float[length + 1];
                h_y[length] = float.NaN;

                Shader shader = new BlockTranspose(block, n, m);

                CudaArray<float> d_x = h_x;
                CudaArray<float> d_y = h_y;

                shader.Execute(stream: null, d_x, d_y);

                d_y.Read(h_y);

                for (int k = 0, idx = 0; k < n; k++) {
                    for (int j = 0; j < m; j++) {
                        float expected = block * (k + n * j);

                        for (int i = 0; i < block; i++, idx++) { 
                            Assert.AreEqual(expected, h_y[idx]);
                            expected++;
                        }
                    }    
                }

                Assert.IsTrue(float.IsNaN(h_y[length]));

                Console.WriteLine(shader);
            }
        }
    }
}
