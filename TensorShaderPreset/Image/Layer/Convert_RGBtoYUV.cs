﻿using TensorShader;
using static TensorShader.Field;

namespace TensorShaderPreset.Image {
    /// <summary>色空間変換</summary>
    public static partial class Convert {

        /// <summary>RGB->YUV(PAL, SECAM)</summary>
        public static Field RGBtoYUV(Field x) {
            float[] wval =
                {
                   0.299f,       0.587f,       0.114f,
                  -0.14714119f, -0.28886916f, +0.43601035f,
                  +0.61497538f, -0.51496512f, -0.10001026f
                };

            VariableField w = new Tensor(Shape.Kernel0D(3, 3), wval);

            return PointwiseConvolution2D(x, w);
        }

        /// <summary>YUV->RGB(PAL, SECAM)</summary>
        public static Field YUVtoRGB(Field x) {
            float[] wval =
                {
                  1,  0,            +1.139883030f,
                  1, -0.394642334f, -0.580621850f,
                  1, +2.032061853f,  0
                };

            VariableField w = new Tensor(Shape.Kernel0D(3, 3), wval);

            return PointwiseConvolution2D(x, w);
        }
    }
}
