using TensorShader;
using static TensorShader.Field;

namespace TensorShaderPreset.Image {
    /// <summary>フィルタ</summary>
    public static partial class Filter {

        /// <summary>X方向微分フィルタ</summary>
        public static NdimArray<float> KernelDiffX(int inchannels, float scale) {
            float[] kernel =
                {
                    0,      0, 0,
                    +scale, 0, -scale,
                    0,      0, 0,
                };

            return KernelSpatialFilter(inchannels, kwidth: 3, kheight: 3, kernel);
        }

        /// <summary>X方向微分フィルタ</summary>
        public static Field DiffX(Field x, float scale) {
            VariableField w = new VariableField(
                KernelDiffX(x.Shape.Channels, scale),
                name: "DiffX"
            );

            return ChannelwiseConvolution2D(x, w);
        }

        /// <summary>Y方向微分フィルタ</summary>
        public static NdimArray<float> KernelDiffY(int inchannels, float scale) {
            float[] kernel =
                {
                   0, +scale, 0,
                   0, 0,      0,
                   0, -scale, 0,
                };

            return KernelSpatialFilter(inchannels, kwidth: 3, kheight: 3, kernel);
        }

        /// <summary>Y方向微分フィルタ</summary>
        public static Field DiffY(Field x, float scale) {
            VariableField w = new VariableField(
                KernelDiffY(x.Shape.Channels, scale),
                name: "DiffY"
            );

            return ChannelwiseConvolution2D(x, w);
        }
    }
}
