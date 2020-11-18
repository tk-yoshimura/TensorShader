using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using TensorShaderCudaBackend.API;

namespace TensorShaderCudaBackendTest.APITest {
    [TestClass]
    public class CudaRuntimeTest {
        [TestMethod]
        public void DevicePropTest() {
            int device_counts = Cuda.DeviceCounts;

            Console.WriteLine($"DeviceCounts : {device_counts}");
            Console.WriteLine($"CurrentDeviceID : {Cuda.CurrectDeviceID}");

            for (int device_id = 0; device_id < device_counts; device_id++) {
                Console.WriteLine($"DeviceID : {device_id}");

                Cuda.DeviceProp prop = Cuda.DeviceProperty(device_id);

                Console.WriteLine($"    Name : {prop.Name}");
                Console.WriteLine($"    UUID : {string.Join("-", prop.UUID)}");
                Console.WriteLine($"    LUID : {string.Join("-", prop.LUID)}");
                Console.WriteLine($"    LuidDeviceNodeMask : {prop.LuidDeviceNodeMask}");
                Console.WriteLine($"    GlobalMemoryBytes : {prop.GlobalMemoryBytes}");
                Console.WriteLine($"    SharedMemoryBytesPerBlock : {prop.SharedMemoryBytesPerBlock}");
                Console.WriteLine($"    RegisterSizePerBlock : {prop.RegisterSizePerBlock}");
                Console.WriteLine($"    WarpSize : {prop.WarpSize}");
                Console.WriteLine($"    MemoryPitchBytes : {prop.MemoryPitchBytes}");
                Console.WriteLine($"    MaxThreadsPerBlock : {prop.MaxThreadsPerBlock}");
                Console.WriteLine($"    MaxThreadsDim : {string.Join(",", prop.MaxThreadsDim)}");
                Console.WriteLine($"    MaxGridsDim : {string.Join(",", prop.MaxGridsDim)}");
                Console.WriteLine($"    ClockRate : {prop.ClockRate}");
                Console.WriteLine($"    ConstMemoryBytes : {prop.ConstMemoryBytes}");
                Console.WriteLine($"    ComputeCapability : {(prop.Major, prop.Minor)}");
                Console.WriteLine($"    TextureAlignment : {prop.TextureAlignment}");
                Console.WriteLine($"    TexturePitchAlignment : {prop.TexturePitchAlignment}");
                Console.WriteLine($"    DeviceOverlap : {prop.DeviceOverlap}");
                Console.WriteLine($"    MultiProcessorCount : {prop.MultiProcessorCount}");
                Console.WriteLine($"    KernelExecTimeout : {prop.KernelExecTimeout}");
                Console.WriteLine($"    Integrated : {prop.Integrated}");
                Console.WriteLine($"    CanMapHostMemory : {prop.CanMapHostMemory}");
                Console.WriteLine($"    ComputeMode : {prop.ComputeMode}");
                Console.WriteLine($"    ConcurrentKernels : {prop.ConcurrentKernels}");
                Console.WriteLine($"    ECCSupport : {prop.ECCSupport}");
                Console.WriteLine($"    PciBusID : {prop.PciBusID}");
                Console.WriteLine($"    PciDeviceID : {prop.PciDeviceID}");
                Console.WriteLine($"    PciDomainID : {prop.PciDomainID}");
                Console.WriteLine($"    MemoryClockRate : {prop.MemoryClockRate}");
                Console.WriteLine($"    MemoryBusWidth : {prop.MemoryBusWidth}");
                Console.WriteLine($"    L2CacheBytes : {prop.L2CacheBytes}");
                Console.WriteLine($"    MaxThreadsPerMultiProcessor : {prop.MaxThreadsPerMultiProcessor}");
            }
        }

        [TestMethod]
        public void AllocateTest() {
            IntPtr ptr = Cuda.Memory.Allocate<float>(15);

            Assert.AreNotEqual(IntPtr.Zero, ptr);

            Cuda.Memory.Free(ref ptr);

            Assert.AreEqual(IntPtr.Zero, ptr);

            Console.WriteLine($"meminfo: {Cuda.MemoryInfo}");
            Console.WriteLine($"memusage: {Cuda.MemoryUsage}");
        }

        [TestMethod]
        public void MemcpyTest() {
            const int count = 15;

            float[] v = (new float[count]).Select((_, idx) => (float)idx).ToArray();
            float[] v2 = new float[count];

            IntPtr ptr = Cuda.Memory.Allocate<float>(count);
            IntPtr ptr2 = Cuda.Memory.Allocate<float>(count);

            Cuda.Memory.CopyHostToDevice(v, ptr, count);
            Cuda.Memory.CopyDeviceToDevice<float>(ptr, ptr2, count);
            Cuda.Memory.CopyDeviceToHost(ptr2, v2, count);

            CollectionAssert.AreEqual(v, v2);

            Cuda.Memory.Free(ref ptr);
            Cuda.Memory.Free(ref ptr2);
        }

        [TestMethod]
        public void MemcpyWithIndexTest() {
            const int count = 15, index = 3;

            /*write*/ { 
                float[] v = (new float[count]).Select((_, idx) => (float)idx).ToArray();
                float[] v2 = new float[count];
                float[] v3 = (new float[count]).Select((_, idx) => idx < count - index ? (float)idx + index : 0).ToArray();

                IntPtr ptr = Cuda.Memory.Allocate<float>(count);
                IntPtr ptr2 = Cuda.Memory.Allocate<float>(count);

                Cuda.Memory.Zeroset<float>(ptr, count);
                Cuda.Memory.Zeroset<float>(ptr2, count);

                Cuda.Memory.CopyHostToDevice(v, ptr, count - index, index);
                Cuda.Memory.CopyDeviceToDevice<float>(ptr, ptr2, count);
                Cuda.Memory.CopyDeviceToHost(ptr2, v2, count);

                CollectionAssert.AreEqual(v3, v2);

                Cuda.Memory.Free(ref ptr);
                Cuda.Memory.Free(ref ptr2);
            }

            /*read*/ { 
                float[] v = (new float[count]).Select((_, idx) => (float)idx).ToArray();
                float[] v2 = new float[count];
                float[] v3 = (new float[count]).Select((_, idx) => idx < index ? 0 : (float)idx - index).ToArray();

                IntPtr ptr = Cuda.Memory.Allocate<float>(count);
                IntPtr ptr2 = Cuda.Memory.Allocate<float>(count);

                Cuda.Memory.Zeroset<float>(ptr, count);
                Cuda.Memory.Zeroset<float>(ptr2, count);

                Cuda.Memory.CopyHostToDevice(v, ptr, count);
                Cuda.Memory.CopyDeviceToDevice<float>(ptr, ptr2, count);
                Cuda.Memory.CopyDeviceToHost(ptr2, v2, count - index, index);

                CollectionAssert.AreEqual(v3, v2);

                Cuda.Memory.Free(ref ptr);
                Cuda.Memory.Free(ref ptr2);
            }
        }

        [TestMethod]
        public void ZerosetTest() {
            const int count = 15;

            float[] v = (new float[count]).Select((_, idx) => (float)idx).ToArray();
            float[] v2 = (new float[count]).Select((_, idx) => idx < (float)(count - 1) ? 0 : (float)idx).ToArray();

            IntPtr ptr = Cuda.Memory.Allocate<float>(count);

            Cuda.Memory.CopyHostToDevice(v, ptr, count);

            Cuda.Memory.Zeroset<float>(ptr, count - 1);

            Cuda.Memory.CopyDeviceToHost(ptr, v, count);

            CollectionAssert.AreEqual(v2, v);

            Cuda.Memory.Free(ref ptr);
        }
    }
}
