using System;
using System.Linq;

namespace TensorShaderTest.Operators.Complex {
    public class ComplexMap0D {
        private readonly System.Numerics.Complex[] val;

        public int Channels { private set; get; }
        public int Batch { private set; get; }
        public int Length => Channels * Batch;

        public ComplexMap0D(int channels, int batch, System.Numerics.Complex[] val = null) {
            if (channels < 1 || batch < 1) {
                throw new ArgumentException();
            }

            int length = checked(channels * batch);

            if (!(val is null) && val.Length != length) {
                throw new ArgumentException(nameof(val));
            }

            this.val = (val is null) ? new System.Numerics.Complex[length] : (System.Numerics.Complex[])val.Clone();
            this.Channels = channels;
            this.Batch = batch;
        }

        public System.Numerics.Complex this[int ch, int th] {
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

        public System.Numerics.Complex this[int idx] {
            get {
                return val[idx];
            }
            set {
                val[idx] = value;
            }
        }

        public static bool operator ==(ComplexMap0D map1, ComplexMap0D map2) {
            if (map1.Channels != map2.Channels) return false;
            if (map1.Batch != map2.Batch) return false;

            return map1.val.SequenceEqual(map2.val);
        }

        public static bool operator !=(ComplexMap0D map1, ComplexMap0D map2) {
            return !(map1 == map2);
        }

        public override bool Equals(object obj) {
            return obj is ComplexMap0D map && this == map;
        }

        public override int GetHashCode() {
            return base.GetHashCode();
        }

        public float[] ToArray() {
            return (new float[Length * 2]).Select((_, idx) => idx % 2 == 0 ? (float)val[idx / 2].Real : (float)val[idx / 2].Imaginary).ToArray();
        }
    }
}
