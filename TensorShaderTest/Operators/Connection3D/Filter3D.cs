using System;
using System.Linq;

namespace TensorShaderTest.Operators.Connection3D {
    public class Filter3D {
        private readonly double[] val;

        public int InChannels { private set; get; }
        public int OutChannels { private set; get; }
        public int KernelWidth { private set; get; }
        public int KernelHeight { private set; get; }
        public int KernelDepth { private set; get; }
        public int Length => InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth;

        public Filter3D(int inchannels, int outchannels, int kwidth, int kheight, int kdepth, float[] val = null) {
            if (kwidth < 1 || kheight < 1 || kdepth < 1 || inchannels < 1 || outchannels < 1) {
                throw new ArgumentException();
            }

            int length = checked(inchannels * outchannels * kwidth * kheight * kdepth);

            if (!(val is null) && val.Length != length) {
                throw new ArgumentException(null, nameof(val));
            }

            this.val = (val is null) ? new double[length] : val.Select((v) => (double)v).ToArray();
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;
            this.InChannels = inchannels;
            this.OutChannels = outchannels;
        }

        public double this[int inch, int outch, int kx, int ky, int kz] {
            get {
                if (kx < 0 || kx >= KernelWidth || ky < 0 || ky >= KernelHeight || kz < 0 || kz >= KernelDepth || inch < 0 || inch >= InChannels || outch < 0 || outch >= OutChannels) {
                    throw new IndexOutOfRangeException();
                }

                return val[inch + InChannels * (outch + OutChannels * (kx + KernelWidth * (ky + KernelHeight * kz)))];
            }
            set {
                if (kx < 0 || kx >= KernelWidth || ky < 0 || ky >= KernelHeight || kz < 0 || kz >= KernelDepth || inch < 0 || inch >= InChannels || outch < 0 || outch >= OutChannels) {
                    throw new IndexOutOfRangeException();
                }

                val[inch + InChannels * (outch + OutChannels * (kx + KernelWidth * (ky + KernelHeight * kz)))] = value;
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

        public static bool operator ==(Filter3D filter1, Filter3D filter2) {
            if (filter1.KernelWidth != filter2.KernelWidth) return false;
            if (filter1.KernelHeight != filter2.KernelHeight) return false;
            if (filter1.KernelDepth != filter2.KernelDepth) return false;
            if (filter1.InChannels != filter2.InChannels) return false;
            if (filter1.OutChannels != filter2.OutChannels) return false;

            return filter1.val.SequenceEqual(filter2.val);
        }

        public static bool operator !=(Filter3D filter1, Filter3D filter2) {
            return !(filter1 == filter2);
        }

        public override bool Equals(object obj) {
            return obj is Filter3D filter && this == filter;
        }

        public override int GetHashCode() {
            return base.GetHashCode();
        }

        public float[] ToArray() {
            return val.Select((v) => (float)v).ToArray();
        }
    }
}
