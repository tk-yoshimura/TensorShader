using TensorShader;

namespace TensorShaderPreset.Image {
    /// <summary>フィルタ</summary>
    public static partial class Filter {

        /// <summary>X方向ソーベルフィルタ</summary>
        public static Field SobelX(Field x, float scale) {
            float[] kernel =
                {
                   +0.25f * scale, 0, -0.25f * scale,
                   +0.50f * scale, 0, -0.50f * scale,
                   +0.25f * scale, 0, -0.25f * scale,
                };

            return SpatialFilter(x, kwidth: 3, kheight: 3, kernel, name: "SobelX");
        }

        /// <summary>Y方向ソーベルフィルタ</summary>
        public static Field SobelY(Field x, float scale) {
            float[] kernel =
                {
                   +0.25f * scale, +0.50f * scale, +0.25f * scale,
                    0,              0,              0,
                   -0.25f * scale, -0.50f * scale, -0.25f * scale,
                };

            return SpatialFilter(x, kwidth: 3, kheight: 3, kernel, name: "SobelY");
        }
    }
}
