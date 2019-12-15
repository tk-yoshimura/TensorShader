using System;
using System.IO.Compression;
using System.IO;
using System.Linq;
using TensorShader;
using TensorShaderUtil.BatchGenerator;

namespace MNIST {
    public class MnistImagesLoader : MultitaskBatchGenerator {
        protected byte[] filedata;

        private static readonly int mnist_size = 28, mnist_length = mnist_size * mnist_size;

        public int Count { private set; get; }

        public MnistImagesLoader(string filepath, int num_batches, int count)
            : base(Shape.Map2D(channels: 1, width: mnist_size, height: mnist_size), num_batches) {
            byte[] filedata = null;

            using (var stream = new FileStream(filepath, FileMode.Open)) {
                using (var gz_stream = new GZipStream(stream, CompressionMode.Decompress)) {
                    using (var memory_stream = new MemoryStream()) {
                        gz_stream.CopyTo(memory_stream);

                        filedata = memory_stream.ToArray();
                    }
                }
            }

            this.Count = count;

            this.filedata = filedata;
        }

        public override float[] GenerateData(int index) {
            if (index < 0 || index >= Count) {
                throw new IndexOutOfRangeException(nameof(index));
            }

            byte[] data = new byte[mnist_length];

            Buffer.BlockCopy(filedata, index * mnist_length + 16, data, 0, mnist_length);

            return data.Select((b) => b / 255f).ToArray();
        }
    }
}
