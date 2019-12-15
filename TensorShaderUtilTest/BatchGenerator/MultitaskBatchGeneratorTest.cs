using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShaderUtil.BatchGenerator;

namespace TensorShaderUtilTest.BatchGenerator {
    [TestClass]
    public class MultitaskBatchGeneratorTest {
        [TestMethod]
        public void ExecuteTest() {
            int num_batches = 1024, channels = 5;

            IBatchGenerator generator = new TestMultitaskGenerator(channels, num_batches);

            int[] indexes = (new int[num_batches]).Select((_, idx) => idx).ToArray();

            generator.Request(indexes);

            float[] value = generator.Receive();

            CollectionAssert.AreEqual((new int[channels * num_batches]).Select((_, idx) => (float)idx).ToArray(), value);

            generator.Request();

            float[] value2 = generator.Receive();

            CollectionAssert.AreEqual((new int[channels * num_batches]).Select((_, idx) => (float)(idx % 5)).ToArray(), value2);
        }

        [TestMethod]
        public void InvalidOperationTest() {
            int num_batches = 1024, channels = 5;

            IBatchGenerator generator = new TestMultitaskGenerator(channels, num_batches);

            int[] indexes = (new int[num_batches]).Select((_, idx) => idx).ToArray();

            Assert.ThrowsException<InvalidOperationException>(
                () => { float[] value = generator.Receive(); }
            );
        }

        public class TestMultitaskGenerator : MultitaskBatchGenerator {
            public int Channels { private set; get; }

            public TestMultitaskGenerator(int channels, int num_batches)
                : base(Shape.Vector(channels), num_batches) {
                this.Channels = channels;
            }

            public override float[] GenerateData(int index) {
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
