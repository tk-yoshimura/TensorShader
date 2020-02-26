using TensorShader;

namespace TensorShaderPreset.Image {
    /// <summary>フィルタ</summary>
    public static partial class Filter {

        /// <summary>X方向微分フィルタ</summary>
        public static Field DiffX(Field x, float scale) {
            float[] kernel =
                {
                    0,      0, 0,
                    +scale, 0, -scale,
                    0,      0, 0,
                };

            return SpatialFilter(x, kwidth: 3, kheight: 3, kernel, name: "DiffX");
        }

        /// <summary>Y方向微分フィルタ</summary>
        public static Field DiffY(Field x, float scale) {
            float[] kernel =
                {
                   0, +scale, 0,
                   0, 0,      0,
                   0, -scale, 0,
                };

            return SpatialFilter(x, kwidth: 3, kheight: 3, kernel, name: "DiffY");
        }
    }
}
