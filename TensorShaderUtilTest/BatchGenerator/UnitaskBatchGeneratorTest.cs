using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShaderUtil.BatchGenerator;

namespace TensorShaderUtilTest.BatchGenerator {
    [TestClass]
    public class UnitaskBatchGeneratorTest {
        [TestMethod]
        public void ExecuteTest() {
            int num_batches = 1024, channels = 5;

            IBatchGenerator generator = new TestUnitaskGenerator(channels, num_batches);

            int[] indexes = (new int[num_batches]).Select((_, idx) => idx).ToArray();

            generator.Request(indexes);

            NdimArray<float> value = generator.Receive();

            CollectionAssert.AreEqual((new int[channels * num_batches]).Select((_, idx) => (float)idx).ToArray(), value.Value);

            generator.Request();

            NdimArray<float> value2 = generator.Receive();

            CollectionAssert.AreEqual((new int[channels * num_batches]).Select((_, idx) => (float)(idx % 5)).ToArray(), value2.Value);
        }

        [TestMethod]
        public void InvalidOperationTest() {
            int num_batches = 1024, channels = 5;

            IBatchGenerator generator = new TestUnitaskGenerator(channels, num_batches);

            int[] indexes = (new int[num_batches]).Select((_, idx) => idx).ToArray();

            Assert.ThrowsException<InvalidOperationException>(
                () => { NdimArray<float> value = generator.Receive(); }
            );

        }

        public class TestUnitaskGenerator : UnitaskBatchGenerator {
            public int Channels { private set; get; }

            public TestUnitaskGenerator(int channels, int num_batches)
                : base(Shape.Vector(channels), num_batches) {
                this.Channels = channels;
            }

            public override NdimArray<float> GenerateData(int index) {
                return new float[] {
                    index * Channels,
                    index * Channels + 1,
                    index * Channels + 2,
                    index * Channels + 3,
                    index * Channels + 4
                };
            }
        }
    }
}
