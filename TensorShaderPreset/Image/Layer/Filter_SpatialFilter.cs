using System;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderPreset.Image {
    /// <summary>フィルタ</summary>
    public static partial class Filter {

        /// <summary>空間フィルタ</summary>
        public static NdimArray<float> KernelSpatialFilter(int inchannels, int kwidth, int kheight, float[] kernel) {
            if (kwidth * kheight != kernel.Length) {
                throw new ArgumentException(nameof(kernel));
            }

            float[] wval = new float[inchannels * kernel.Length];

            for (int s = 0; s < kernel.Length; s++) {
                for (int ch = 0; ch < inchannels; ch++) {
                    wval[ch + inchannels * s] = kernel[s];
                }
            }

            return new NdimArray<float>(Shape.Kernel2D(inchannels, 1, kwidth, kheight), wval);
        }

        /// <summary>空間フィルタ</summary>
        public static Field SpatialFilter(Field x, int kwidth, int kheight, float[] kernel, string name) {
            VariableField w = new VariableField(
                KernelSpatialFilter(x.Shape.Channels, kwidth, kheight, kernel), 
                name
            );

            return ChannelwiseConvolution2D(x, w);
        }
    }
}
