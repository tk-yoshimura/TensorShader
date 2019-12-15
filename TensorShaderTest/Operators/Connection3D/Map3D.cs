using System;
using System.Linq;

namespace TensorShaderTest.Operators.Connection3D {
    public class Map3D {
        private readonly double[] val;

        public int Channels { private set; get; }
        public int Width { private set; get; }
        public int Height { private set; get; }
        public int Depth { private set; get; }
        public int Batch { private set; get; }
        public int Length => Channels * Width * Height * Depth * Batch;

        public Map3D(int channels, int width, int height, int depth, int batch, float[] val = null) {
            if (width < 1 || height < 1 || depth < 1 || channels < 1 || batch < 1) {
                throw new ArgumentException();
            }

            int length = checked(width * height * depth * channels * batch);

            if (!(val is null) && val.Length != length) {
                throw new ArgumentException(nameof(val));
            }

            this.val = (val is null) ? new double[length] : val.Select((v) => (double)v).ToArray();
            this.Width = width;
            this.Height = height;
            this.Depth = depth;
            this.Channels = channels;
            this.Batch = batch;
        }

        public double this[int ch, int x, int y, int z, int th] {
            get {
                if (x < 0 || x >= Width || y < 0 || y >= Height || z < 0 || z >= Depth || ch < 0 || ch >= Channels || th < 0 || th >= Batch) {
                    throw new IndexOutOfRangeException();
                }

                return val[ch + Channels * (x + Width * (y + Height * (z + Depth * th)))];
            }
            set {
                if (x < 0 || x >= Width || y < 0 || y >= Height || z < 0 || z >= Depth || ch < 0 || ch >= Channels || th < 0 || th >= Batch) {
                    throw new IndexOutOfRangeException();
                }

                val[ch + Channels * (x + Width * (y + Height * (z + Depth * th)))] = value;
            }
        }

        public double this[int idx] {
            get {
                return val[idx];
            }
            set {
                val[idx] = value;
            }
        }

        public static bool operator ==(Map3D map1, Map3D map2) {
            if (map1.Width != map2.Width) return false;
            if (map1.Height != map2.Height) return false;
            if (map1.Depth != map2.Depth) return false;
            if (map1.Channels != map2.Channels) return false;
            if (map1.Batch != map2.Batch) return false;

            return map1.val.SequenceEqual(map2.val);
        }

        public static bool operator !=(Map3D map1, Map3D map2) {
            return !(map1 == map2);
        }

        public override bool Equals(object obj) {
            return obj is Map3D map && this == map;
        }

        public override int GetHashCode() {
            return base.GetHashCode();
        }

        public float[] ToArray() {
            return val.Select((v) => (float)v).ToArray();
        }
    }
}
