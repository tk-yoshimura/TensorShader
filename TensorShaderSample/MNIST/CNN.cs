using TensorShader;
using TensorShader.Layers;
using static TensorShader.Field;

namespace MNIST {
    public static class CNN {
        public static Field Forward(Field x, int classes) {
            Convolution2D conv1 =
                new Convolution2D(
                    inchannels: 1, outchannels: 4,
                    kwidth: 3, kheight: 3,
                    use_bias: true,
                    pad_mode: PaddingMode.Zero, label: "conv1");

            Convolution2D conv2 =
                new Convolution2D(
                    inchannels: 4, outchannels: 8,
                    kwidth: 3, kheight: 3,
                    use_bias: true,
                    pad_mode: PaddingMode.Zero, label: "conv2");

            Convolution2D conv3 =
                new Convolution2D(
                    inchannels: 8, outchannels: 16,
                    kwidth: 3, kheight: 3,
                    use_bias: true,
                    pad_mode: PaddingMode.Zero, label: "conv3");

            Field h1 = Relu(conv1.Forward(x));
            Field h2 = MaxPooling2D(h1, stride: 2);

            Field h3 = Relu(conv2.Forward(h2));
            Field h4 = MaxPooling2D(h3, stride: 2);

            Field h5 = Relu(conv3.Forward(h4));

            Dense fc =
                new Dense(
                    inchannels: h5.Shape.DataSize, outchannels: classes,
                    use_bias: true, label: "fc");

            Field y = fc.Forward(h5);

            return y;
        }
    }
}
