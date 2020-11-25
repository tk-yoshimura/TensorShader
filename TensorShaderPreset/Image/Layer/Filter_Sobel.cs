using TensorShader;
using static TensorShader.Field;

namespace TensorShaderPreset.Image {
    /// <summary>フィルタ</summary>
    public static partial class Filter {

        /// <summary>X方向ソーベルフィルタ</summary>
        public static NdimArray<float> KernelSobelX(int inchannels, float scale) { 
            float[] kernel =
                {
                   +0.25f * scale, 0, -0.25f * scale,
                   +0.50f * scale, 0, -0.50f * scale,
                   +0.25f * scale, 0, -0.25f * scale,
                };

            return KernelSpatialFilter(inchannels, kwidth: 3, kheight: 3, kernel);
        }

        /// <summary>X方向ソーベルフィルタ</summary>
        public static Field SobelX(Field x, float scale) {
            VariableField w = new VariableField(
                KernelSobelX(x.Shape.Channels, scale), 
                name: "SobelX"
            );

            return ChannelwiseConvolution2D(x, w);
        }

        /// <summary>Y方向ソーベルフィルタ</summary>
        public static NdimArray<float> KernelSobelY(int inchannels, float scale) { 
            float[] kernel =
                {
                   +0.25f * scale, +0.50f * scale, +0.25f * scale,
                    0,              0,              0,
                   -0.25f * scale, -0.50f * scale, -0.25f * scale,
                };

            return KernelSpatialFilter(inchannels, kwidth: 3, kheight: 3, kernel);
        }

        /// <summary>Y方向ソーベルフィルタ</summary>
        public static Field SobelY(Field x, float scale) {
            VariableField w = new VariableField(
                KernelSobelY(x.Shape.Channels, scale), 
                name: "SobelY"
            );

            return ChannelwiseConvolution2D(x, w);
        }
    }
}
