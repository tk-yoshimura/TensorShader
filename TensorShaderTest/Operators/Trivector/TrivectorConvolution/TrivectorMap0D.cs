using System;
using System.Linq;

namespace TensorShaderTest.Operators.Trivector {
    public class TrivectorMap0D {
        private readonly Trivector[] val;

        public int Channels { private set; get; }
        public int Batch { private set; get; }
        public int Length => Channels * Batch;

        public TrivectorMap0D(int channels, int batch, Trivector[] val = null) {
            if (channels < 1 || batch < 1) {
                throw new ArgumentException();
            }

            int length = checked(channels * batch);

            if (!(val is null) && val.Length != length) {
                throw new ArgumentException(null, nameof(val));
            }

            this.val = (val is null) ? new Trivector[length] : (Trivector[])val.Clone();
            this.Channels = channels;
            this.Batch = batch;
        }

        public Trivector this[int ch, int th] {
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

        public Trivector this[int idx] {
            get {
                return val[idx];
            }
            set {
                val[idx] = value;
            }
        }

        public static bool operator ==(TrivectorMap0D map1, TrivectorMap0D map2) {
            if (map1.Channels != map2.Channels) return false;
            if (map1.Batch != map2.Batch) return false;

            return map1.val.SequenceEqual(map2.val);
        }

        public static bool operator !=(TrivectorMap0D map1, TrivectorMap0D map2) {
            return !(map1 == map2);
        }

        public override bool Equals(object obj) {
            return obj is TrivectorMap0D map && this == map;
        }

        public override int GetHashCode() {
            return base.GetHashCode();
        }

        public float[] ToArray() {
            return (new float[Length * 3]).Select((_, idx) => (float)val[idx / 3][idx % 3]).ToArray();
        }
    }
}
