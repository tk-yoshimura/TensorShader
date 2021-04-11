using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace TensorShaderTest.Operators.Trivector {
    public struct Trivector {
        public double X { set; get; }
        public double Y { set; get; }
        public double Z { set; get; }

        public Trivector(double x, double y, double z) {
            this.X = x;
            this.Y = y;
            this.Z = z;
        }

        public double this[int index] {
            get {
                if (index == 0) {
                    return X;
                }
                else if (index == 1) {
                    return Y;
                }
                else if (index == 2) {
                    return Z;
                }
                throw new ArgumentException(nameof(index));
            }
        }

        public static Trivector operator +(Trivector v1, Trivector v2) {
            return new Trivector(v1.X + v2.X, v1.Y + v2.Y, v1.Z + v2.Z);
        }

        public static Trivector operator *(Trivector v, Quaternion.Quaternion q) {
            double x = v.X, y = v.Y, z = v.Z, r = q.R, i = q.I, j = q.J, k = q.K;

            return new Trivector(
                x * (r * r + i * i - j * j - k * k) + 2 * y * (i * j - r * k) + 2 * z * (k * i + r * j),
                y * (r * r - i * i + j * j - k * k) + 2 * z * (j * k - r * i) + 2 * x * (i * j + r * k),
                z * (r * r - i * i - j * j + k * k) + 2 * x * (k * i - r * j) + 2 * y * (j * k + r * i)
            );
        }

        public static Trivector MulVGrad(Trivector v, Quaternion.Quaternion q) {
            double x = v.X, y = v.Y, z = v.Z, r = q.R, i = q.I, j = q.J, k = q.K;

            return new Trivector(
                x * (r * r + i * i - j * j - k * k) + 2 * y * (i * j + r * k) + 2 * z * (k * i - r * j),
                y * (r * r - i * i + j * j - k * k) + 2 * z * (j * k + r * i) + 2 * x * (i * j - r * k),
                z * (r * r - i * i - j * j + k * k) + 2 * x * (k * i + r * j) + 2 * y * (j * k - r * i)
            );
        }

        public static Quaternion.Quaternion MulQGrad(Trivector v, Trivector u, Quaternion.Quaternion q) {
            double vx = v.X, vy = v.Y, vz = v.Z, ux = u.X, uy = u.Y, uz = u.Z, r = q.R, i = q.I, j = q.J, k = q.K;

            return new Quaternion.Quaternion(
                ux * (2 * j * vz - 2 * k * vy + 2 * r * vx) + uy * (2 * k * vx - 2 * i * vz + 2 * r * vy) + uz * (2 * i * vy - 2 * j * vx + 2 * r * vz),
                ux * (2 * k * vz + 2 * j * vy + 2 * i * vx) + uy * (2 * j * vx - 2 * r * vz - 2 * i * vy) + uz * (2 * r * vy + 2 * k * vx - 2 * i * vz),
                ux * (2 * r * vz + 2 * i * vy - 2 * j * vx) + uy * (2 * i * vx + 2 * k * vz + 2 * j * vy) + uz * (2 * k * vy - 2 * r * vx - 2 * j * vz),
                ux * (2 * i * vz - 2 * r * vy - 2 * k * vx) + uy * (2 * r * vx + 2 * j * vz - 2 * k * vy) + uz * (2 * j * vy + 2 * i * vx + 2 * k * vz)
            );
        }
    }

    [TestClass]
    public class TrivectorTest {
        [TestMethod]
        public void AddTest() {
            Trivector v1 = new(2, 7, 31);
            Trivector v2 = new(23, 17, 3);

            Trivector v_add = v1 + v2;

            Assert.AreEqual(25, v_add.X);
            Assert.AreEqual(24, v_add.Y);
            Assert.AreEqual(34, v_add.Z);
        }

        [TestMethod]
        public void MulTest() {
            Trivector v = new(4, 3, 2);
            Quaternion.Quaternion q = new(5, -6, 7, -8);

            Trivector u = v * q;

            Assert.AreEqual(112, u.X);
            Assert.AreEqual(-838, u.Y);
            Assert.AreEqual(-404, u.Z);
        }

        [TestMethod]
        public void MulVGradTest() {
            Trivector v = new(4, 3, 2);
            Quaternion.Quaternion q = new(5, -6, 7, -8);

            Trivector u = Trivector.MulVGrad(v, q);

            Assert.AreEqual(-648, u.X);
            Assert.AreEqual(-438, u.Y);
            Assert.AreEqual(516, u.Z);
        }

        [TestMethod]
        public void MulQGradTest() {
            Trivector v = new(4, -3, 2);
            Trivector u = new(-2, 5, -7);
            Quaternion.Quaternion q = new(5, -6, 7, -8);

            Quaternion.Quaternion p = Trivector.MulQGrad(v, u, q);

            Assert.AreEqual(-390, p.R);
            Assert.AreEqual(734, p.I);
            Assert.AreEqual(-470, p.J);
            Assert.AreEqual(814, p.K);
        }
    }
}
