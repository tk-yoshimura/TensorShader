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
            for(uint length = 1; length < 100; length++) { 

                CudaArray<float> arr = new CudaArray<float>(length);

                Assert.IsTrue(arr.IsValid);
                Assert.AreEqual(Marshal.SizeOf(typeof(float)) * length, (int)arr.ByteSize);
                Assert.AreEqual((ulong)length, arr.Length);

                arr.Dispose();

                Assert.IsFalse(arr.IsValid);
                Assert.AreEqual(0ul, arr.ByteSize);
            }

            for(uint length = 1; length < 100; length++) { 

                CudaArray<double> arr = new CudaArray<double>(length);

                Assert.IsTrue(arr.IsValid);
                Assert.AreEqual(Marshal.SizeOf(typeof(double)) * length, (int)arr.ByteSize);
                Assert.AreEqual((ulong)length, arr.Length);

                arr.Dispose();

                Assert.IsFalse(arr.IsValid);
                Assert.AreEqual(0ul, arr.ByteSize);
            }
        }

        [TestMethod]
        public void InitializeTest() {
            for(uint length = 1; length < 100; length++) { 

                float[] v = new float[length];

                CudaArray<float> arr = new CudaArray<float>(length, zeroset: false);

                arr.Read(v);

                foreach (float f in v) {
                    Console.WriteLine(f);
                }
            }

            for(uint length = 1; length < 100; length++) { 

                double[] v = new double[length];

                CudaArray<double> arr = new CudaArray<double>(length, zeroset: false);

                arr.Read(v);

                foreach (double f in v) {
                    Console.WriteLine(f);
                }
            }
        }

        [TestMethod]
        public void WriteReadTest() {
            for(uint length = 1; length < 100; length++) { 

                float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
                float[] v2 = new float[length];

                CudaArray<float> arr = new CudaArray<float>(length);
                CudaArray<float> arr2 = new CudaArray<float>(length);

                arr.Write(v);

                CudaArray<float>.Copy(arr, arr2, length);

                arr2.Read(v2);

                CollectionAssert.AreEqual(v, v2);
            }

            for(uint length = 1; length < 100; length++) { 

                double[] v = (new double[length]).Select((_, idx) => (double)idx).ToArray();
                double[] v2 = new double[length];

                CudaArray<double> arr = new CudaArray<double>(length);
                CudaArray<double> arr2 = new CudaArray<double>(length);

                arr.Write(v);

                CudaArray<double>.Copy(arr, arr2, length);

                arr2.Read(v2);

                CollectionAssert.AreEqual(v, v2);
            }
        }

        [TestMethod]
        public void WriteReadAsyncTest() {
            Stream stream = new Stream();

            for(uint length = 1; length < 100; length++) { 

                float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
                float[] v2 = new float[length];

                CudaArray<float> arr = new CudaArray<float>(length);
                CudaArray<float> arr2 = new CudaArray<float>(length);

                arr.Write(v);

                CudaArray<float>.CopyAsync(stream, arr, arr2, length);

                arr2.Read(v2);

                CollectionAssert.AreEqual(v, v2);
            }

            for(uint length = 1; length < 100; length++) { 

                double[] v = (new double[length]).Select((_, idx) => (double)idx).ToArray();
                double[] v2 = new double[length];

                CudaArray<double> arr = new CudaArray<double>(length);
                CudaArray<double> arr2 = new CudaArray<double>(length);

                arr.Write(v);

                CudaArray<double>.CopyAsync(stream, arr, arr2, length);

                arr2.Read(v2);

                CollectionAssert.AreEqual(v, v2);
            }
        }

        [TestMethod]
        public void RegionWriteReadTest() {
            for(uint length = 1; length < 100; length++) { 

                float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
                float[] v2 = new float[length];
                float[] v3 = (new float[length]).Select((_, idx) => idx < length - 1 ? (float)idx : 0).ToArray();

                CudaArray<float> arr = new CudaArray<float>(length);
                CudaArray<float> arr2 = new CudaArray<float>(length);

                arr.Write(v, length - 1);

                CudaArray<float>.Copy(arr, arr2, length);

                arr2.Read(v2, length - 1);

                CollectionAssert.AreEqual(v2, v3);
            }

            for(uint length = 1; length < 100; length++) { 

                double[] v = (new double[length]).Select((_, idx) => (double)idx).ToArray();
                double[] v2 = new double[length];
                double[] v3 = (new double[length]).Select((_, idx) => idx < length - 1 ? (double)idx : 0).ToArray();

                CudaArray<double> arr = new CudaArray<double>(length);
                CudaArray<double> arr2 = new CudaArray<double>(length);

                arr.Write(v, length - 1);

                CudaArray<double>.Copy(arr, arr2, length);

                arr2.Read(v2, length - 1);

                CollectionAssert.AreEqual(v2, v3);
            }
        }

        [TestMethod]
        public void RegionWriteReadAsyncTest() {
            Stream stream = new Stream();

            for(uint length = 1; length < 100; length++) { 

                float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
                float[] v2 = new float[length];
                float[] v3 = (new float[length]).Select((_, idx) => idx < length - 1 ? (float)idx : 0).ToArray();

                CudaArray<float> arr = new CudaArray<float>(length);
                CudaArray<float> arr2 = new CudaArray<float>(length);

                arr.Write(v, length - 1);

                CudaArray<float>.CopyAsync(stream, arr, arr2, length);

                arr2.Read(v2, length - 1);

                CollectionAssert.AreEqual(v2, v3);
            }

            for(uint length = 1; length < 100; length++) { 

                double[] v = (new double[length]).Select((_, idx) => (double)idx).ToArray();
                double[] v2 = new double[length];
                double[] v3 = (new double[length]).Select((_, idx) => idx < length - 1 ? (double)idx : 0).ToArray();

                CudaArray<double> arr = new CudaArray<double>(length);
                CudaArray<double> arr2 = new CudaArray<double>(length);

                arr.Write(v, length - 1);

                CudaArray<double>.CopyAsync(stream, arr, arr2, length);

                arr2.Read(v2, length - 1);

                CollectionAssert.AreEqual(v2, v3);
            }
        }

        [TestMethod]
        public void ZerosetTest() {
            for(uint length = 1; length < 100; length++) { 
                for(uint index = 0; index < length; index++) { 
                    for(uint count = 0; count <= length - index; count++) {
                        float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
                        float[] v2 = (new float[length])
                            .Select((_, idx) => idx >= index && idx < (index + count) ? 0 : (float)idx)
                            .ToArray();
                        
                        CudaArray<float> arr = new CudaArray<float>(v);

                        arr.Zeroset(index, count);

                        float[] v3 = arr;

                        CollectionAssert.AreEqual(v2, v3);
                    }
                }
            }

            {
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

            for(uint length = 1; length < 100; length++) { 
                for(uint index = 0; index < length; index++) { 
                    for(uint count = 0; count <= length - index; count++) {
                        double[] v = (new double[length]).Select((_, idx) => (double)idx).ToArray();
                        double[] v2 = (new double[length])
                            .Select((_, idx) => idx >= index && idx < (index + count) ? 0 : (double)idx)
                            .ToArray();
                        
                        CudaArray<double> arr = new CudaArray<double>(v);

                        arr.Zeroset(index, count);

                        double[] v3 = arr;

                        CollectionAssert.AreEqual(v2, v3);
                    }
                }
            }

            {
                const int length = 15;

                double[] v = (new double[length]).Select((_, idx) => (double)idx).ToArray();
                double[] v2 = new double[length];
                double[] v3 = (new double[length]).Select((_, idx) => idx < (double)(length - 1) ? 0 : (double)idx).ToArray();
                double[] v4 = (new double[length]).Select((_, idx) => idx >= 3 && idx < (double)(length - 1) ? 0 : (double)idx).ToArray();

                double[] v5 = new double[length];

                CudaArray<double> arr = new CudaArray<double>(length);

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
        }

        [TestMethod]
        public void ZerosetAsyncTest() {
            Stream stream = new Stream();

            for(uint length = 1; length < 100; length++) { 
                for(uint index = 0; index < length; index++) { 
                    for(uint count = 0; count <= length - index; count++) {
                        float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
                        float[] v2 = (new float[length])
                            .Select((_, idx) => idx >= index && idx < (index + count) ? 0 : (float)idx)
                            .ToArray();
                        
                        CudaArray<float> arr = new CudaArray<float>(v);

                        arr.ZerosetAsync(stream, index, count);

                        float[] v3 = arr;

                        CollectionAssert.AreEqual(v2, v3);
                    }
                }
            }

            {
                const int length = 15;

                float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
                float[] v2 = new float[length];
                float[] v3 = (new float[length]).Select((_, idx) => idx < (float)(length - 1) ? 0 : (float)idx).ToArray();
                float[] v4 = (new float[length]).Select((_, idx) => idx >= 3 && idx < (float)(length - 1) ? 0 : (float)idx).ToArray();

                float[] v5 = new float[length];

                CudaArray<float> arr = new CudaArray<float>(length);

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

            for(uint length = 1; length < 100; length++) { 
                for(uint index = 0; index < length; index++) { 
                    for(uint count = 0; count <= length - index; count++) {
                        double[] v = (new double[length]).Select((_, idx) => (double)idx).ToArray();
                        double[] v2 = (new double[length])
                            .Select((_, idx) => idx >= index && idx < (index + count) ? 0 : (double)idx)
                            .ToArray();
                        
                        CudaArray<double> arr = new CudaArray<double>(v);

                        arr.ZerosetAsync(stream, index, count);

                        double[] v3 = arr;

                        CollectionAssert.AreEqual(v2, v3);
                    }
                }
            }

            {
                const int length = 15;

                double[] v = (new double[length]).Select((_, idx) => (double)idx).ToArray();
                double[] v2 = new double[length];
                double[] v3 = (new double[length]).Select((_, idx) => idx < (double)(length - 1) ? 0 : (double)idx).ToArray();
                double[] v4 = (new double[length]).Select((_, idx) => idx >= 3 && idx < (double)(length - 1) ? 0 : (double)idx).ToArray();

                double[] v5 = new double[length];

                CudaArray<double> arr = new CudaArray<double>(length);

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
        }

        [TestMethod]
        public void CopyTest() {
            for(uint src_length = 1; src_length <= 16; src_length++) { 
                for(uint dst_length = 1; dst_length <= 16; dst_length++) { 
                    for(uint src_index = 0; src_index < src_length; src_index++) { 
                        for(uint dst_index = 0; dst_index < dst_length; dst_index++) { 
                            for(uint count = 0; count <= Math.Min(src_length - src_index, dst_length - dst_index); count++) {

                                float[] v = (new float[src_length]).Select((_, idx) => (float)idx).ToArray();
                                float[] v2 = (new float[dst_length])
                                    .Select((_, idx) => idx >= dst_index && idx < (dst_index + count) ? v[idx - dst_index + src_index] : 0)
                                    .ToArray();
                        
                                CudaArray<float> arr_src = new CudaArray<float>(v);
                                CudaArray<float> arr_dst = new CudaArray<float>(dst_length);

                                CudaArray<float>.Copy(arr_src, src_index, arr_dst, dst_index, count);

                                float[] v3 = arr_dst;

                                CollectionAssert.AreEqual(v2, v3);

                            }
                        }
                    }
                }
            }

            {
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

            for(uint src_length = 1; src_length <= 16; src_length++) { 
                for(uint dst_length = 1; dst_length <= 16; dst_length++) { 
                    for(uint src_index = 0; src_index < src_length; src_index++) { 
                        for(uint dst_index = 0; dst_index < dst_length; dst_index++) { 
                            for(uint count = 0; count <= Math.Min(src_length - src_index, dst_length - dst_index); count++) {

                                double[] v = (new double[src_length]).Select((_, idx) => (double)idx).ToArray();
                                double[] v2 = (new double[dst_length])
                                    .Select((_, idx) => idx >= dst_index && idx < (dst_index + count) ? v[idx - dst_index + src_index] : 0)
                                    .ToArray();
                        
                                CudaArray<double> arr_src = new CudaArray<double>(v);
                                CudaArray<double> arr_dst = new CudaArray<double>(dst_length);

                                CudaArray<double>.Copy(arr_src, src_index, arr_dst, dst_index, count);

                                double[] v3 = arr_dst;

                                CollectionAssert.AreEqual(v2, v3);

                            }
                        }
                    }
                }
            }

            {
                const int length = 15;

                double[] v = (new double[length]).Select((_, idx) => (double)idx).ToArray();
                double[] v3 = (new double[length]).Select((_, idx) => idx < (double)(length - 1) ? (double)idx : 0).ToArray();
                double[] v4 = (new double[length]).Select((_, idx) => idx >= 3 && idx < (double)(length - 1) ? (double)idx : 0).ToArray();
                double[] v5 = (new double[length]).Select((_, idx) => idx >= 2 && idx < (double)(length - 2) ? (double)idx + 1 : 0).ToArray();

                double[] v6 = new double[length];

                CudaArray<double> arr = new CudaArray<double>(v);
                CudaArray<double> arr2 = new CudaArray<double>(length);

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
        }

        [TestMethod]
        public void CopyAsyncTest() {
            Stream stream = new Stream();

            for(uint src_length = 1; src_length <= 16; src_length++) { 
                for(uint dst_length = 1; dst_length <= 16; dst_length++) { 
                    for(uint src_index = 0; src_index < src_length; src_index++) { 
                        for(uint dst_index = 0; dst_index < dst_length; dst_index++) { 
                            for(uint count = 0; count <= Math.Min(src_length - src_index, dst_length - dst_index); count++) {

                                float[] v = (new float[src_length]).Select((_, idx) => (float)idx).ToArray();
                                float[] v2 = (new float[dst_length])
                                    .Select((_, idx) => idx >= dst_index && idx < (dst_index + count) ? v[idx - dst_index + src_index] : 0)
                                    .ToArray();
                        
                                CudaArray<float> arr_src = new CudaArray<float>(v);
                                CudaArray<float> arr_dst = new CudaArray<float>(dst_length);

                                CudaArray<float>.CopyAsync(stream, arr_src, src_index, arr_dst, dst_index, count);

                                float[] v3 = arr_dst;

                                CollectionAssert.AreEqual(v2, v3);

                            }
                        }
                    }
                }
            }

            {
                const int length = 15;

                float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
                float[] v3 = (new float[length]).Select((_, idx) => idx < (float)(length - 1) ? (float)idx : 0).ToArray();
                float[] v4 = (new float[length]).Select((_, idx) => idx >= 3 && idx < (float)(length - 1) ? (float)idx : 0).ToArray();
                float[] v5 = (new float[length]).Select((_, idx) => idx >= 2 && idx < (float)(length - 2) ? (float)idx + 1 : 0).ToArray();

                float[] v6 = new float[length];

                CudaArray<float> arr = new CudaArray<float>(v);
                CudaArray<float> arr2 = new CudaArray<float>(length);

                arr2.Zeroset();
                arr.CopyToAsync(stream, arr2, arr.Length);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v, v6);

                arr2.Zeroset();
                arr.CopyToAsync(stream, arr2, arr.Length - 1);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v3, v6);

                arr2.Zeroset();
                arr.CopyToAsync(stream, 3, arr2, 3, arr.Length - 4);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v4, v6);

                arr2.Zeroset();
                arr.CopyToAsync(stream, 3, arr2, 2, arr.Length - 4);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v5, v6);
            }

            for(uint src_length = 1; src_length <= 16; src_length++) { 
                for(uint dst_length = 1; dst_length <= 16; dst_length++) { 
                    for(uint src_index = 0; src_index < src_length; src_index++) { 
                        for(uint dst_index = 0; dst_index < dst_length; dst_index++) { 
                            for(uint count = 0; count <= Math.Min(src_length - src_index, dst_length - dst_index); count++) {

                                double[] v = (new double[src_length]).Select((_, idx) => (double)idx).ToArray();
                                double[] v2 = (new double[dst_length])
                                    .Select((_, idx) => idx >= dst_index && idx < (dst_index + count) ? v[idx - dst_index + src_index] : 0)
                                    .ToArray();
                        
                                CudaArray<double> arr_src = new CudaArray<double>(v);
                                CudaArray<double> arr_dst = new CudaArray<double>(dst_length);

                                CudaArray<double>.CopyAsync(stream, arr_src, src_index, arr_dst, dst_index, count);

                                double[] v3 = arr_dst;

                                CollectionAssert.AreEqual(v2, v3);

                            }
                        }
                    }
                }
            }

            {
                const int length = 15;

                double[] v = (new double[length]).Select((_, idx) => (double)idx).ToArray();
                double[] v3 = (new double[length]).Select((_, idx) => idx < (double)(length - 1) ? (double)idx : 0).ToArray();
                double[] v4 = (new double[length]).Select((_, idx) => idx >= 3 && idx < (double)(length - 1) ? (double)idx : 0).ToArray();
                double[] v5 = (new double[length]).Select((_, idx) => idx >= 2 && idx < (double)(length - 2) ? (double)idx + 1 : 0).ToArray();

                double[] v6 = new double[length];

                CudaArray<double> arr = new CudaArray<double>(v);
                CudaArray<double> arr2 = new CudaArray<double>(length);

                arr2.Zeroset();
                arr.CopyToAsync(stream, arr2, arr.Length);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v, v6);

                arr2.Zeroset();
                arr.CopyToAsync(stream, arr2, arr.Length - 1);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v3, v6);

                arr2.Zeroset();
                arr.CopyToAsync(stream, 3, arr2, 3, arr.Length - 4);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v4, v6);

                arr2.Zeroset();
                arr.CopyToAsync(stream, 3, arr2, 2, arr.Length - 4);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v5, v6);
            }
        }

        [TestMethod]
        public void BadCreateTest() {
            Assert.ThrowsException<TypeInitializationException>(() => { 
                CudaArray<char> arr = new CudaArray<char>(32);
            });
        }
    }
}
