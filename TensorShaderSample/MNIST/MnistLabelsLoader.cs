using System;
using System.IO.Compression;
using System.IO;
using TensorShaderUtil.BatchGenerator;
using TensorShader;

namespace MNIST {
    public class MnistLabelsLoader : UnitaskBatchGenerator {
        protected byte[] filedata;

        public int Count { private set; get; }

        public MnistLabelsLoader(string filepath, int num_batches, int count)
            : base(Shape.Scalar(), num_batches) {
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

            float label = filedata[index + 8];

            return new float[] { label };
        }
    }
}
