using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace TensorShaderTest {
    public static class AssertError {
        public static void Tolerance(float expected, float actual, float minerr, float rateerr) {
            float delta = Math.Max(minerr, Math.Max(Math.Abs(expected), Math.Abs(actual)) * rateerr);

            Assert.AreEqual(expected, actual, delta);
        }

        public static void Tolerance(float expected, float actual, float minerr, float rateerr, string message) {
            float delta = Math.Max(minerr, Math.Max(Math.Abs(expected), Math.Abs(actual)) * rateerr);

            Assert.AreEqual(expected, actual, delta, message);
        }

        public static void Tolerance(double expected, double actual, double minerr, double rateerr) {
            double delta = Math.Max(minerr, Math.Max(Math.Abs(expected), Math.Abs(actual)) * rateerr);

            Assert.AreEqual(expected, actual, delta);
        }

        public static void Tolerance(double expected, double actual, double minerr, double rateerr, string message) {
            double delta = Math.Max(minerr, Math.Max(Math.Abs(expected), Math.Abs(actual)) * rateerr);

            Assert.AreEqual(expected, actual, delta, message);
        }

        public static void Tolerance(float[] expecteds, float[] actuals, float minerr, float rateerr) {
            Assert.AreEqual(expecteds.Length, actuals.Length, "length");

            for (int i = 0; i < expecteds.Length; i++) {
                float expected = expecteds[i], actual = actuals[i];

                Tolerance(expected, actual, minerr, rateerr);
            }
        }

        public static void Tolerance(float[] expecteds, float[] actuals, float minerr, float rateerr, string message) {
            Assert.AreEqual(expecteds.Length, actuals.Length, $"{message} length");

            for (int i = 0; i < expecteds.Length; i++) {
                float expected = expecteds[i], actual = actuals[i];

                Tolerance(expected, actual, minerr, rateerr, $"{message} {i}");
            }
        }

        public static void Tolerance(double[] expecteds, double[] actuals, double minerr, double rateerr) {
            Assert.AreEqual(expecteds.Length, actuals.Length, "length");

            for (int i = 0; i < expecteds.Length; i++) {
                double expected = expecteds[i], actual = actuals[i];

                Tolerance(expected, actual, minerr, rateerr, $"{i}");
            }
        }

        public static void Tolerance(double[] expecteds, double[] actuals, double minerr, double rateerr, string message) {
            Assert.AreEqual(expecteds.Length, actuals.Length, $"{message} length");

            for (int i = 0; i < expecteds.Length; i++) {
                double expected = expecteds[i], actual = actuals[i];

                Tolerance(expected, actual, minerr, rateerr, $"{message} {i}");
            }
        }

        public static void Tolerance(float[] expecteds, float[] actuals, float minerr, float rateerr, ref float max_err) {
            Assert.AreEqual(expecteds.Length, actuals.Length, "length");

            for (int i = 0; i < expecteds.Length; i++) {
                float expected = expecteds[i], actual = actuals[i];

                Tolerance(expected, actual, minerr, rateerr, $"{i}");

                max_err = Math.Max(max_err, Math.Abs(expected - actual));
            }
        }

        public static void Tolerance(float[] expecteds, float[] actuals, float minerr, float rateerr, ref float max_err, string message) {
            Assert.AreEqual(expecteds.Length, actuals.Length, $"{message} length");

            for (int i = 0; i < expecteds.Length; i++) {
                float expected = expecteds[i], actual = actuals[i];

                Tolerance(expected, actual, minerr, rateerr, $"{message} {i}");

                max_err = Math.Max(max_err, Math.Abs(expected - actual));
            }
        }

        public static void Tolerance(double[] expecteds, double[] actuals, double minerr, ref double max_err, double rateerr) {
            Assert.AreEqual(expecteds.Length, actuals.Length, "length");

            for (int i = 0; i < expecteds.Length; i++) {
                double expected = expecteds[i], actual = actuals[i];

                Tolerance(expected, actual, minerr, rateerr, $"{i}");

                max_err = Math.Max(max_err, Math.Abs(expected - actual));
            }
        }

        public static void Tolerance(double[] expecteds, double[] actuals, double minerr, double rateerr, ref double max_err, string message) {
            Assert.AreEqual(expecteds.Length, actuals.Length, $"{message} length");

            for (int i = 0; i < expecteds.Length; i++) {
                double expected = expecteds[i], actual = actuals[i];

                Tolerance(expected, actual, minerr, rateerr, $"{message} {i}");

                max_err = Math.Max(max_err, Math.Abs(expected - actual));
            }
        }
    }

    [TestClass]
    public class AssertErrorTest {
        [TestMethod]
        public void ExecuteTest() {
            AssertError.Tolerance(0, 0, 1e-3, 1e-5);
            AssertError.Tolerance(0, 1e-4, 1e-3, 1e-5);
            AssertError.Tolerance(1e-4, 0, 1e-3, 1e-5);
            AssertError.Tolerance(1e-4, 1e-4, 1e-3, 1e-5);
            AssertError.Tolerance(1e-2, 9.9999e-3, 1e-6, 1e-5);

            Assert.ThrowsException<AssertFailedException>(() =>
                AssertError.Tolerance(0, 2e-3, 1e-3, 1e-5)
            );

            Assert.ThrowsException<AssertFailedException>(() =>
                AssertError.Tolerance(2e-3, 0, 1e-3, 1e-5)
            );

            Assert.ThrowsException<AssertFailedException>(() =>
                AssertError.Tolerance(1e-3, 3e-3, 1e-3, 1e-5)
            );

            Assert.ThrowsException<AssertFailedException>(() =>
                AssertError.Tolerance(1e-2, 9.998e-3, 1e-6, 1e-5)
            );
        }
    }
}
