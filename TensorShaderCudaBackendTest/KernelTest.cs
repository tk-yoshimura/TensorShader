using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShaderCudaBackend;
using TensorShaderCudaBackend.API;

namespace TensorShaderCudaBackendTest {
    [TestClass]
    public class KernelTest {
        readonly string code = @"
        __global__ void add(float *x1, float *x2, float *y, int length) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= length) {
                return;
            }
            y[i] = x1[i] + x2[i];
        }";

        [TestMethod]
        public void CreateTest() {

            Kernel kernel = new(code, "add");

            Assert.IsTrue(kernel.IsValid);

            kernel.Dispose();

            Assert.IsFalse(kernel.IsValid);
        }

        [TestMethod]
        public void ThreadDimsTest() {

            Assert.AreEqual(512u, Kernel.DefaultBlockSize(65536));
            Assert.AreEqual((1u, 1u), Kernel.DefaultBlockSize((1, 1)));
            Assert.AreEqual((512u, 1u), Kernel.DefaultBlockSize((65536, 65536)));
            Assert.AreEqual((1u, 512u), Kernel.DefaultBlockSize((1, 65536)));
            Assert.AreEqual((512u, 1u), Kernel.DefaultBlockSize((65536, 1)));
            Assert.AreEqual((512u, 1u, 1u), Kernel.DefaultBlockSize((2048, 1, 1)));
            Assert.AreEqual((1u, 512u, 1u), Kernel.DefaultBlockSize((1, 2048, 1)));
            Assert.AreEqual((1u, 1u, 64u), Kernel.DefaultBlockSize((1, 1, 2048)));
            Assert.AreEqual((32u, 16u, 1u), Kernel.DefaultBlockSize((32, 32, 1)));
            Assert.AreEqual((1u, 32u, 16u), Kernel.DefaultBlockSize((1, 32, 32)));
            Assert.AreEqual((32u, 1u, 16u), Kernel.DefaultBlockSize((32, 1, 32)));
        }

        [TestMethod]
        public void VectorAddTest() {
            const int length = 20000;
            Random random = new(1324);

            float[] h_a = (new float[length]).Select((_) => (float)random.NextDouble()).ToArray();
            float[] h_b = (new float[length]).Select((_) => (float)random.NextDouble()).ToArray();
            float[] h_c = new float[length];

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "vectoradd.nvvp");
            Cuda.Profiler.Start();

            CudaArray<float> d_a = new(h_a);
            CudaArray<float> d_b = new(h_b);
            CudaArray<float> d_c = new(length);

            Kernel kernel = new(code, "add");

            kernel.Execute(length, dynamic_shared_memory_bytes: 0, stream: null, d_a, d_b, d_c, length); // d_a=a, d_b=b, d_c=a+b
            kernel.Execute(length, dynamic_shared_memory_bytes: 0, stream: null, d_c, d_b, d_a, length); // d_a=a+2b, d_b=b, d_c=a+b
            kernel.Execute(length, dynamic_shared_memory_bytes: 0, stream: null, d_a, d_b, d_c, length); // d_a=a+2b, d_b=b, d_c=a+3b

            d_c.Read(h_c);

            Cuda.Profiler.Stop();

            for (int i = 0; i < length; i++) {
                Assert.AreEqual(h_a[i] + 3 * h_b[i], h_c[i], 1e-6f);
            }
        }

        [TestMethod]
        public void VectorAddAsyncTest() {
            const int length = 20000;
            Random random = new(1324);

            float[] h_a = (new float[length]).Select((_) => (float)random.NextDouble()).ToArray();
            float[] h_b = (new float[length]).Select((_) => (float)random.NextDouble()).ToArray();
            float[] h_c = new float[length];

            Kernel kernel = new(code, "add");

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "vectoraddasync.nvvp");
            Cuda.Profiler.Start();

            CudaArray<float> d_a = new(h_a);
            CudaArray<float> d_b = new(h_b);
            CudaArray<float> d_c = new(length);

            Stream stream = new();

            kernel.Execute(length, dynamic_shared_memory_bytes: 0, stream, d_a, d_b, d_c, length); // d_a=a, d_b=b, d_c=a+b
            kernel.Execute(length, dynamic_shared_memory_bytes: 0, stream, d_c, d_b, d_a, length); // d_a=a+2b, d_b=b, d_c=a+b
            kernel.Execute(length, dynamic_shared_memory_bytes: 0, stream, d_a, d_b, d_c, length); // d_a=a+2b, d_b=b, d_c=a+3b

            d_c.Read(h_c);

            Cuda.Profiler.Stop();

            for (int i = 0; i < length; i++) {
                Assert.AreEqual(h_a[i] + 3 * h_b[i], h_c[i], 1e-6f);
            }
        }
    }
}
