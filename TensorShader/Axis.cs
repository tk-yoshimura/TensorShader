namespace TensorShader {
    /// <summary>軸</summary>
    public static class Axis {
        /// <summary>スカラー</summary>
        public static class Scalar { }

        /// <summary>ベクター</summary>
        public static class Vector {
            /// <summary>チャネル</summary>
            public static int Channels => 0;

            /// <summary>バッチ</summary>
            public static int Batch => 0;
        }

        /// <summary>0次元マップ</summary>
        public static class Map0D {
            /// <summary>チャネル</summary>
            public static int Channels => 0;

            /// <summary>バッチ</summary>
            public static int Batch => 1;
        }

        /// <summary>1次元マップ</summary>
        public static class Map1D {
            /// <summary>チャネル</summary>
            public static int Channels => 0;

            /// <summary>幅</summary>
            public static int Width => 1;

            /// <summary>バッチ</summary>
            public static int Batch => 2;
        }

        /// <summary>2次元マップ</summary>
        public static class Map2D {
            /// <summary>チャネル</summary>
            public static int Channels => 0;

            /// <summary>幅</summary>
            public static int Width => 1;

            /// <summary>高さ</summary>
            public static int Height => 2;

            /// <summary>バッチ</summary>
            public static int Batch => 3;
        }

        /// <summary>3次元マップ</summary>
        public static class Map3D {
            /// <summary>チャネル</summary>
            public static int Channels => 0;

            /// <summary>幅</summary>
            public static int Width => 1;

            /// <summary>高さ</summary>
            public static int Height => 2;

            /// <summary>奥行き</summary>
            public static int Depth => 3;

            /// <summary>バッチ</summary>
            public static int Batch => 4;
        }

        /// <summary>0次元フィルタ</summary>
        public static class Kernel0D {
            /// <summary>入力チャネル</summary>
            public static int InChannels => 0;

            /// <summary>出力チャネル</summary>
            public static int OutChannels => 1;
        }

        /// <summary>1次元フィルタ</summary>
        public static class Kernel1D {
            /// <summary>入力チャネル</summary>
            public static int InChannels => 0;

            /// <summary>出力チャネル</summary>
            public static int OutChannels => 1;

            /// <summary>幅</summary>
            public static int Width => 2;
        }

        /// <summary>2次元フィルタ</summary>
        public static class Kernel2D {
            /// <summary>入力チャネル</summary>
            public static int InChannels => 0;

            /// <summary>出力チャネル</summary>
            public static int OutChannels => 1;

            /// <summary>幅</summary>
            public static int Width => 2;

            /// <summary>高さ</summary>
            public static int Height => 3;
        }

        /// <summary>3次元フィルタ</summary>
        public static class Kernel3D {
            /// <summary>入力チャネル</summary>
            public static int InChannels => 0;

            /// <summary>出力チャネル</summary>
            public static int OutChannels => 1;

            /// <summary>幅</summary>
            public static int Width => 2;

            /// <summary>高さ</summary>
            public static int Height => 3;

            /// <summary>奥行き</summary>
            public static int Depth => 4;
        }

        /// <summary>混同行列</summary>
        public static class ConfusionMatrix {
            /// <summary>予測</summary>
            public static int Actual => 0;

            /// <summary>正解</summary>
            public static int Expect => 1;
        }
    }
}
