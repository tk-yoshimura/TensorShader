using System;
using TensorShader;

namespace MNIST {
    public class MnistLoader {
        private readonly MnistImagesLoader train_images, test_images;
        private readonly MnistLabelsLoader train_labels, test_labels;

        public int CountTrainDatas =>
            (train_images.Count == train_labels.Count)
            ? train_images.Count
            : throw new Exception("Mismatch counts");

        public int CountTestDatas =>
            (test_images.Count == test_labels.Count)
            ? test_images.Count
            : throw new Exception("Mismatch counts");

        public int NumBatches => train_images.NumBatches;

        public Shape DataShape => train_images.DataShape;

        public Shape BatchShape => train_images.BatchShape;

        public MnistLoader(string dirpath, int num_batches) {
            this.train_images = new MnistImagesLoader(dirpath + "/train-images-idx3-ubyte.gz", num_batches, count: 60000);
            this.train_labels = new MnistLabelsLoader(dirpath + "/train-labels-idx1-ubyte.gz", num_batches, count: 60000);
            this.test_images = new MnistImagesLoader(dirpath + "/t10k-images-idx3-ubyte.gz", num_batches, count: 10000);
            this.test_labels = new MnistLabelsLoader(dirpath + "/t10k-labels-idx1-ubyte.gz", num_batches, count: 10000);
        }

        public (float[] images, float[] labels) GetTrain(int[] indexes) {
            float[] images = train_images.Get(indexes);
            float[] labels = train_labels.Get(indexes);

            return (images, labels);
        }

        public (float[] images, float[] labels) GetTest(int[] indexes) {
            float[] images = test_images.Get(indexes);
            float[] labels = test_labels.Get(indexes);

            return (images, labels);
        }
    }
}
