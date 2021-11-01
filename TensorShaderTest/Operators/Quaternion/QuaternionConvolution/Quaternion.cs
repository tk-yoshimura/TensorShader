using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace TensorShaderTest.Operators.Quaternion {
    public struct Quaternion {
        public double R { set; get; }
        public double I { set; get; }
        public double J { set; get; }
        public double K { set; get; }

        public Quaternion(double r, double i, double j, double k) {
            this.R = r;
            this.I = i;
            this.J = j;
            this.K = k;
        }

        public double this[int index] {
            get {
                if (index == 0) {
                    return R;
                }
                else if (index == 1) {
                    return I;
                }
                else if (index == 2) {
                    return J;
                }
                else if (index == 3) {
                    return K;
                }
                throw new ArgumentException(null, nameof(index));
            }
        }

        public static Quaternion operator +(Quaternion q1, Quaternion q2) {
            return new Quaternion(q1.R + q2.R, q1.I + q2.I, q1.J + q2.J, q1.K + q2.K);
        }

        public static Quaternion operator *(Quaternion q1, Quaternion q2) {
            return new Quaternion(
                q1.R * q2.R - q1.I * q2.I - q1.J * q2.J - q1.K * q2.K,
                q1.R * q2.I + q1.I * q2.R + q1.J * q2.K - q1.K * q2.J,
                q1.R * q2.J - q1.I * q2.K + q1.J * q2.R + q1.K * q2.I,
                q1.R * q2.K + q1.I * q2.J - q1.J * q2.I + q1.K * q2.R
            );
        }

        public static implicit operator Quaternion(int v) {
            return new Quaternion(v, 0, 0, 0);
        }

        public static Quaternion MulGrad(Quaternion q1, Quaternion q2) {
            return new Quaternion(
                  q1.R * q2.R + q1.I * q2.I + q1.J * q2.J + q1.K * q2.K,
                -q1.R * q2.I + q1.I * q2.R + q1.J * q2.K - q1.K * q2.J,
                -q1.R * q2.J - q1.I * q2.K + q1.J * q2.R + q1.K * q2.I,
                -q1.R * q2.K + q1.I * q2.J - q1.J * q2.I + q1.K * q2.R
            );
        }
    }

    [TestClass]
    public class QuaternionTest {
        [TestMethod]
        public void AddTest() {
            System.Numerics.Quaternion sq1 = new(2, 3, 4, 1);
            System.Numerics.Quaternion sq2 = new(7, 6, 5, 8);

            Quaternion q1 = new(sq1.W, sq1.X, sq1.Y, sq1.Z);
            Quaternion q2 = new(sq2.W, sq2.X, sq2.Y, sq2.Z);

            System.Numerics.Quaternion sq_add = sq1 + sq2;
            Quaternion q_add = q1 + q2;

            Assert.AreEqual(sq_add.W, q_add.R);
            Assert.AreEqual(sq_add.X, q_add.I);
            Assert.AreEqual(sq_add.Y, q_add.J);
            Assert.AreEqual(sq_add.Z, q_add.K);
        }

        [TestMethod]
        public void MulTest() {
            System.Numerics.Quaternion sq1 = new(2, 3, 4, 1);
            System.Numerics.Quaternion sq2 = new(7, 6, 5, 8);

            Quaternion q1 = new(sq1.W, sq1.X, sq1.Y, sq1.Z);
            Quaternion q2 = new(sq2.W, sq2.X, sq2.Y, sq2.Z);

            System.Numerics.Quaternion sq_mul = sq1 * sq2;
            Quaternion q_mul = q1 * q2;

            Assert.AreEqual(sq_mul.W, q_mul.R);
            Assert.AreEqual(sq_mul.X, q_mul.I);
            Assert.AreEqual(sq_mul.Y, q_mul.J);
            Assert.AreEqual(sq_mul.Z, q_mul.K);
        }
    }
}
