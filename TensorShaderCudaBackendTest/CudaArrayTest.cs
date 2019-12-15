using System;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderCudaBackend;

namespace TensorShaderCudaBackendTest {
    [TestClass]
    public class CudaArrayTest {
        [TestMethod]
        public void CreateTest() {
            const int length = 15;

            CudaArray<float> arr = new CudaArray<float>(length);

            Assert.IsTrue(arr.IsValid);
            Assert.AreEqual(Marshal.SizeOf(typeof(float)) * length, (int)arr.ByteSize);

            arr.Dispose();

            Assert.IsFalse(arr.IsValid);
            Assert.AreEqual(0ul, arr.ByteSize);
        }

        [TestMethod]
        public void InitializeTest() {
            const int length = 15;

            float[] v = new float[length];

            CudaArray<float> arr = new CudaArray<float>(length, zeroset: false);

            arr.Read(v);

            foreach (float f in v) {
                Console.WriteLine(f);
            }
        }

        [TestMethod]
        public void WriteReadTest() {
            const int length = 15;

            float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
            float[] v2 = new float[length];

            CudaArray<float> arr = new CudaArray<float>(length);
            CudaArray<float> arr2 = new CudaArray<float>(length);

            arr.Write(v);

            CudaArray<float>.Copy(arr, arr2, length);

            arr2.Read(v2);

            CollectionAssert.AreEqual(v, v2);
        }

        [TestMethod]
        public void WriteReadAsyncTest() {
            const int length = 15;

            float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
            float[] v2 = new float[length];

            CudaArray<float> arr = new CudaArray<float>(length);
            CudaArray<float> arr2 = new CudaArray<float>(length);

            Stream stream = new Stream();

            arr.Write(v);

            CudaArray<float>.CopyAsync(stream, arr, arr2, length);

            arr2.Read(v2);

            CollectionAssert.AreEqual(v, v2);
        }

        [TestMethod]
        public void RegionWriteReadTest() {
            const int length = 15;

            float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
            float[] v2 = new float[length];
            float[] v3 = (new float[length]).Select((_, idx) => idx < (float)(length - 1) ? (float)idx : 0).ToArray();

            CudaArray<float> arr = new CudaArray<float>(length);
            CudaArray<float> arr2 = new CudaArray<float>(length);

            arr.Write(v, length - 1);

            CudaArray<float>.Copy(arr, arr2, length);

            arr2.Read(v2, length - 1);

            CollectionAssert.AreEqual(v2, v3);
        }

        [TestMethod]
        public void RegionWriteReadAsyncTest() {
            const int length = 15;

            float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
            float[] v2 = new float[length];
            float[] v3 = (new float[length]).Select((_, idx) => idx < (float)(length - 1) ? (float)idx : 0).ToArray();

            CudaArray<float> arr = new CudaArray<float>(length);
            CudaArray<float> arr2 = new CudaArray<float>(length);

            Stream stream = new Stream();

            arr.Write(v, length - 1);

            CudaArray<float>.CopyAsync(stream, arr, arr2, length);

            arr2.Read(v2, length - 1);

            CollectionAssert.AreEqual(v2, v3);
        }

        [TestMethod]
        public void ZerosetTest() {
            const int length = 15;

            float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
            float[] v2 = new float[length];
            float[] v3 = (new float[length]).Select((_, idx) => idx < (float)(length - 1) ? 0 : (float)idx).ToArray();
            float[] v4 = (new float[length]).Select((_, idx) => idx >= 3 && idx < (float)(length - 1) ? 0 : (float)idx).ToArray();

            float[] v5 = new float[length];

            CudaArray<float> arr = new CudaArray<float>(length);

            arr.Write(v, length);
            arr.Zeroset();
            arr.Read(v5, length);
            CollectionAssert.AreEqual(v2, v5);

            arr.Write(v, length);
            arr.Zeroset(length - 1);
            arr.Read(v5, length);
            CollectionAssert.AreEqual(v3, v5);

            arr.Write(v, length);
            arr.Zeroset(3, length - 4);
            arr.Read(v5, length);
            CollectionAssert.AreEqual(v4, v5);
        }

        [TestMethod]
        public void ZerosetAsyncTest() {
            const int length = 15;

            float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
            float[] v2 = new float[length];
            float[] v3 = (new float[length]).Select((_, idx) => idx < (float)(length - 1) ? 0 : (float)idx).ToArray();
            float[] v4 = (new float[length]).Select((_, idx) => idx >= 3 && idx < (float)(length - 1) ? 0 : (float)idx).ToArray();

            float[] v5 = new float[length];

            CudaArray<float> arr = new CudaArray<float>(length);

            Stream stream = new Stream();

            arr.Write(v, length);
            arr.ZerosetAsync(stream);
            arr.Read(v5, length);
            CollectionAssert.AreEqual(v2, v5);

            arr.Write(v, length);
            arr.ZerosetAsync(stream, length - 1);
            arr.Read(v5, length);
            CollectionAssert.AreEqual(v3, v5);

            arr.Write(v, length);
            arr.ZerosetAsync(stream, 3, length - 4);
            arr.Read(v5, length);
            CollectionAssert.AreEqual(v4, v5);
        }

        [TestMethod]
        public void CopyTest() {
            const int length = 15;

            float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
            float[] v3 = (new float[length]).Select((_, idx) => idx < (float)(length - 1) ? (float)idx : 0).ToArray();
            float[] v4 = (new float[length]).Select((_, idx) => idx >= 3 && idx < (float)(length - 1) ? (float)idx : 0).ToArray();
            float[] v5 = (new float[length]).Select((_, idx) => idx >= 2 && idx < (float)(length - 2) ? (float)idx + 1 : 0).ToArray();

            float[] v6 = new float[length];

            CudaArray<float> arr = new CudaArray<float>(v);
            CudaArray<float> arr2 = new CudaArray<float>(length);

            arr2.Zeroset();
            arr.CopyTo(arr2, arr.Length);
            arr2.Read(v6);
            CollectionAssert.AreEqual(v, v6);

            arr2.Zeroset();
            arr.CopyTo(arr2, arr.Length - 1);
            arr2.Read(v6);
            CollectionAssert.AreEqual(v3, v6);

            arr2.Zeroset();
            arr.CopyTo(3, arr2, 3, arr.Length - 4);
            arr2.Read(v6);
            CollectionAssert.AreEqual(v4, v6);

            arr2.Zeroset();
            arr.CopyTo(3, arr2, 2, arr.Length - 4);
            arr2.Read(v6);
            CollectionAssert.AreEqual(v5, v6);
        }

        [TestMethod]
        public void CopyAsyncTest() {
            const int length = 15;

            float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
            float[] v3 = (new float[length]).Select((_, idx) => idx < (float)(length - 1) ? (float)idx : 0).ToArray();
            float[] v4 = (new float[length]).Select((_, idx) => idx >= 3 && idx < (float)(length - 1) ? (float)idx : 0).ToArray();
            float[] v5 = (new float[length]).Select((_, idx) => idx >= 2 && idx < (float)(length - 2) ? (float)idx + 1 : 0).ToArray();

            float[] v6 = new float[length];

            CudaArray<float> arr = new CudaArray<float>(v);
            CudaArray<float> arr2 = new CudaArray<float>(length);

            Stream stream = new Stream();

            arr2.ZerosetAsync(stream);
            arr.CopyToAsync(stream, arr2, arr.Length);
            arr2.Read(v6);
            CollectionAssert.AreEqual(v, v6);

            arr2.ZerosetAsync(stream);
            arr.CopyToAsync(stream, arr2, arr.Length - 1);
            arr2.Read(v6);
            CollectionAssert.AreEqual(v3, v6);

            arr2.ZerosetAsync(stream);
            arr.CopyToAsync(stream, 3, arr2, 3, arr.Length - 4);
            arr2.Read(v6);
            CollectionAssert.AreEqual(v4, v6);

            arr2.ZerosetAsync(stream);
            arr.CopyToAsync(stream, 3, arr2, 2, arr.Length - 4);
            arr2.Read(v6);
            CollectionAssert.AreEqual(v5, v6);
        }
    }
}
