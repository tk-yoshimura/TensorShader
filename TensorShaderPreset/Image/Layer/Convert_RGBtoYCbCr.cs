using TensorShader;
using static TensorShader.Field;

namespace TensorShaderPreset.Image {
    /// <summary>色空間変換</summary>
    public static partial class Convert {

        /// <summary>RGB->YCbCr(ITU-R BT.601)</summary>
        public static NdimArray<float> KernelRGBtoYCbCr {
            get {
                const float yr = 0.299f, yg = 0.587f, yb = 0.114f;
                const float cb = 1.772f, cr = 1.402f;

                float[] wval =
                    {
                        +yr,      +yg,      +yb,
                        -yr / cb, -yg / cb, +0.5f,
                        +0.5f,    -yg / cr, -yb / cr
                    };

                return new NdimArray<float>(Shape.Kernel0D(3, 3), wval);
            }
        }

        /// <summary>RGB->YCbCr(ITU-R BT.601)</summary>
        public static Field RGBtoYCbCr(Field x) {
            VariableField w = new VariableField(KernelRGBtoYCbCr, name: "RGBtoYCbCr");

            return PointwiseConvolution2D(x, w);
        }

        /// <summary>YCbCr->RGB(ITU-R BT.601)</summary>
        public static NdimArray<float> KernelYCbCrtoRGB {
            get {
                const float yr = 0.299f, yg = 0.587f, yb = 0.114f;
                const float cb = 1.772f, cr = 1.402f;

                float[] wval =
                {
                    1,  0,            +cr,
                    1, -yb * cb / yg, -yr * cr / yg,
                    1, +cb,            0
                };

                return new NdimArray<float>(Shape.Kernel0D(3, 3), wval);
            }
        }

        /// <summary>YCbCr->RGB(ITU-R BT.601)</summary>
        public static Field YCbCrtoRGB(Field x) {
            VariableField w = new VariableField(KernelYCbCrtoRGB, name: "YCbCrtoRGB");

            return PointwiseConvolution2D(x, w);
        }
    }
}
