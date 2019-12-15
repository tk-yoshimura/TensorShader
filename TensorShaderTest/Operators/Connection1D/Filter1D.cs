using System;
using System.Linq;

namespace TensorShaderTest.Operators.Connection1D {
    public class Filter1D {
        private readonly double[] val;

        public int InChannels { private set; get; }
        public int OutChannels { private set; get; }
        public int KernelWidth { private set; get; }
        public int Length => InChannels * OutChannels * KernelWidth;

        public Filter1D(int inchannels, int outchannels, int kwidth, float[] val = null) {
            if (kwidth < 1 || inchannels < 1 || outchannels < 1) {
                throw new ArgumentException();
            }

            int length = checked(inchannels * outchannels * kwidth);

            if (!(val is null) && val.Length != length) {
                throw new ArgumentException(nameof(val));
            }

            this.val = (val is null) ? new double[length] : val.Select((v) => (double)v).ToArray();
            this.KernelWidth = kwidth;
            this.InChannels = inchannels;
            this.OutChannels = outchannels;
        }

        public double this[int inch, int outch, int kx] {
            get {
                if (kx < 0 || kx >= KernelWidth || inch < 0 || inch >= InChannels || outch < 0 || outch >= OutChannels) {
                    throw new IndexOutOfRangeException();
                }

                return val[inch + InChannels * (outch + OutChannels * kx)];
            }
            set {
                if (kx < 0 || kx >= KernelWidth || inch < 0 || inch >= InChannels || outch < 0 || outch >= OutChannels) {
                    throw new IndexOutOfRangeException();
                }

                val[inch + InChannels * (outch + OutChannels * kx)] = value;
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

        public static bool operator ==(Filter1D filter1, Filter1D filter2) {
            if (filter1.KernelWidth != filter2.KernelWidth) return false;
            if (filter1.InChannels != filter2.InChannels) return false;
            if (filter1.OutChannels != filter2.OutChannels) return false;

            return filter1.val.SequenceEqual(filter2.val);
        }

        public static bool operator !=(Filter1D filter1, Filter1D filter2) {
            return !(filter1 == filter2);
        }

        public override bool Equals(object obj) {
            return obj is Filter1D filter && this == filter;
        }

        public override int GetHashCode() {
            return base.GetHashCode();
        }

        public float[] ToArray() {
            return val.Select((v) => (float)v).ToArray();
        }
    }
}
