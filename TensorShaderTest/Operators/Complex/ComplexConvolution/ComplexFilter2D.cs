using System;
using System.Linq;

namespace TensorShaderTest.Operators.Complex {
    public class ComplexFilter2D {
        private readonly System.Numerics.Complex[] val;

        public int InChannels { private set; get; }
        public int OutChannels { private set; get; }
        public int KernelWidth { private set; get; }
        public int KernelHeight { private set; get; }
        public int Length => InChannels * OutChannels * KernelWidth * KernelHeight;

        public ComplexFilter2D(int inchannels, int outchannels, int kwidth, int kheight, System.Numerics.Complex[] val = null) {
            if (kwidth < 1 || kheight < 1 || inchannels < 1 || outchannels < 1) {
                throw new ArgumentException();
            }

            int length = checked(inchannels * outchannels * kwidth * kheight);

            if (!(val is null) && val.Length != length) {
                throw new ArgumentException(nameof(val));
            }

            this.val = (val is null) ? new System.Numerics.Complex[length] : (System.Numerics.Complex[])val.Clone();
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.InChannels = inchannels;
            this.OutChannels = outchannels;
        }

        public System.Numerics.Complex this[int inch, int outch, int kx, int ky] {
            get {
                if (kx < 0 || kx >= KernelWidth || ky < 0 || ky >= KernelHeight || inch < 0 || inch >= InChannels || outch < 0 || outch >= OutChannels) {
                    throw new IndexOutOfRangeException();
                }

                return val[inch + InChannels * (outch + OutChannels * (kx + KernelWidth * ky))];
            }
            set {
                if (kx < 0 || kx >= KernelWidth || ky < 0 || ky >= KernelHeight || inch < 0 || inch >= InChannels || outch < 0 || outch >= OutChannels) {
                    throw new IndexOutOfRangeException();
                }

                val[inch + InChannels * (outch + OutChannels * (kx + KernelWidth * ky))] = value;
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

        public static bool operator ==(ComplexFilter2D filter1, ComplexFilter2D filter2) {
            if (filter1.KernelWidth != filter2.KernelWidth) return false;
            if (filter1.KernelHeight != filter2.KernelHeight) return false;
            if (filter1.InChannels != filter2.InChannels) return false;
            if (filter1.OutChannels != filter2.OutChannels) return false;

            return filter1.val.SequenceEqual(filter2.val);
        }

        public static bool operator !=(ComplexFilter2D filter1, ComplexFilter2D filter2) {
            return !(filter1 == filter2);
        }

        public override bool Equals(object obj) {
            return obj is ComplexFilter2D filter && this == filter;
        }

        public override int GetHashCode() {
            return base.GetHashCode();
        }

        public float[] ToArray() {
            return (new float[Length * 2]).Select((_, idx) => idx % 2 == 0 ? (float)val[idx / 2].Real : (float)val[idx / 2].Imaginary).ToArray();
        }
    }
}
