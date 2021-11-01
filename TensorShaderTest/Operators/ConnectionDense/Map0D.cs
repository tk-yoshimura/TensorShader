using System;
using System.Linq;

namespace TensorShaderTest.Operators.ConnectionDense {
    public class Map0D {
        private readonly double[] val;

        public int Channels { private set; get; }
        public int Batch { private set; get; }
        public int Length => Channels * Batch;

        public Map0D(int channels, int batch, float[] val = null) {
            if (channels < 1 || batch < 1) {
                throw new ArgumentException();
            }

            int length = checked(channels * batch);

            if (!(val is null) && val.Length != length) {
                throw new ArgumentException(null, nameof(val));
            }

            this.val = (val is null) ? new double[length] : val.Select((v) => (double)v).ToArray();
            this.Channels = channels;
            this.Batch = batch;
        }

        public double this[int ch, int th] {
            get {
                if (ch < 0 || ch >= Channels || th < 0 || th >= Batch) {
                    throw new IndexOutOfRangeException();
                }

                return val[ch + Channels * th];
            }
            set {
                if (ch < 0 || ch >= Channels || th < 0 || th >= Batch) {
                    throw new IndexOutOfRangeException();
                }

                val[ch + Channels * th] = value;
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

        public static bool operator ==(Map0D map1, Map0D map2) {
            if (map1.Channels != map2.Channels) return false;
            if (map1.Batch != map2.Batch) return false;

            return map1.val.SequenceEqual(map2.val);
        }

        public static bool operator !=(Map0D map1, Map0D map2) {
            return !(map1 == map2);
        }

        public override bool Equals(object obj) {
            return obj is Map0D map && this == map;
        }

        public override int GetHashCode() {
            return base.GetHashCode();
        }

        public float[] ToArray() {
            return val.Select((v) => (float)v).ToArray();
        }
    }
}
