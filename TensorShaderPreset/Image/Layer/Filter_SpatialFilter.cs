using System;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderPreset.Image {
    /// <summary>フィルタ</summary>
    public static partial class Filter {

        /// <summary>空間フィルタ</summary>
        public static Field SpatialFilter(Field x, int kwidth, int kheight, float[] kernel, string name) {
            if (kwidth * kheight != kernel.Length) {
                throw new ArgumentException(nameof(kernel));
            }

            int channels = x.Shape.Channels;

            float[] wval = new float[channels * kernel.Length];

            for (int s = 0; s < kernel.Length; s++) {
                for (int ch = 0; ch < channels; ch++) {
                    wval[ch + channels * s] = kernel[s];
                }
            }

            VariableField w = new VariableField(
                (Shape.Kernel2D(channels, 1, kwidth, kheight), wval), 
                name
            );

            return ChannelwiseConvolution2D(x, w);
        }
    }
}
