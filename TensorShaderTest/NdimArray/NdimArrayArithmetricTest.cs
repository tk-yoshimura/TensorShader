using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using TensorShader;

namespace TensorShaderTest {
    [TestClass]
    public class NdimArrayArithmetricTest {
        [TestMethod]
        public void BinaryAddTest() {
            {
                NdimArray<int> arr1 = new int[] { 1, 2, 3 };
                NdimArray<int> arr2 = new int[] { 4, 5, 6 };

                NdimArray<int> arr3 = arr1 + arr2;

                CollectionAssert.AreEqual(new int[] { 5, 7, 9 }, (int[])arr3);
            }

            {
                NdimArray<long> arr1 = new long[] { 1, 2, 3 };
                NdimArray<long> arr2 = new long[] { 4, 5, 6 };

                NdimArray<long> arr3 = arr1 + arr2;

                CollectionAssert.AreEqual(new long[] { 5, 7, 9 }, (long[])arr3);
            }

            {
                NdimArray<float> arr1 = new float[] { 1, 2, 3 };
                NdimArray<float> arr2 = new float[] { 4, 5, 6 };

                NdimArray<float> arr3 = arr1 + arr2;

                CollectionAssert.AreEqual(new float[] { 5, 7, 9 }, (float[])arr3);
            }

            {
                NdimArray<double> arr1 = new double[] { 1, 2, 3 };
                NdimArray<double> arr2 = new double[] { 4, 5, 6 };

                NdimArray<double> arr3 = arr1 + arr2;

                CollectionAssert.AreEqual(new double[] { 5, 7, 9 }, (double[])arr3);
            }

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<short> arr1 = new short[] { 1, 2, 3 };
                NdimArray<short> arr2 = new short[] { 4, 5, 6 };

                NdimArray<short> arr3 = arr1 + arr2;
            });
        }

        [TestMethod]
        public void BinarySubTest() {
            {
                NdimArray<int> arr1 = new int[] { 1, 2, 3 };
                NdimArray<int> arr2 = new int[] { 4, 5, 6 };

                NdimArray<int> arr3 = arr1 - arr2;

                CollectionAssert.AreEqual(new int[] { -3, -3, -3 }, (int[])arr3);
            }

            {
                NdimArray<long> arr1 = new long[] { 1, 2, 3 };
                NdimArray<long> arr2 = new long[] { 4, 5, 6 };

                NdimArray<long> arr3 = arr1 - arr2;

                CollectionAssert.AreEqual(new long[] { -3, -3, -3 }, (long[])arr3);
            }

            {
                NdimArray<float> arr1 = new float[] { 1, 2, 3 };
                NdimArray<float> arr2 = new float[] { 4, 5, 6 };

                NdimArray<float> arr3 = arr1 - arr2;

                CollectionAssert.AreEqual(new float[] { -3, -3, -3 }, (float[])arr3);
            }

            {
                NdimArray<double> arr1 = new double[] { 1, 2, 3 };
                NdimArray<double> arr2 = new double[] { 4, 5, 6 };

                NdimArray<double> arr3 = arr1 - arr2;

                CollectionAssert.AreEqual(new double[] { -3, -3, -3 }, (double[])arr3);
            }

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<short> arr1 = new short[] { 1, 2, 3 };
                NdimArray<short> arr2 = new short[] { 4, 5, 6 };

                NdimArray<short> arr3 = arr1 - arr2;
            });
        }

        [TestMethod]
        public void BinaryMulTest() {
            {
                NdimArray<int> arr1 = new int[] { 1, 2, 3 };
                NdimArray<int> arr2 = new int[] { 4, 5, 6 };

                NdimArray<int> arr3 = arr1 * arr2;

                CollectionAssert.AreEqual(new int[] { 4, 10, 18 }, (int[])arr3);
            }

            {
                NdimArray<long> arr1 = new long[] { 1, 2, 3 };
                NdimArray<long> arr2 = new long[] { 4, 5, 6 };

                NdimArray<long> arr3 = arr1 * arr2;

                CollectionAssert.AreEqual(new long[] { 4, 10, 18 }, (long[])arr3);
            }

            {
                NdimArray<float> arr1 = new float[] { 1, 2, 3 };
                NdimArray<float> arr2 = new float[] { 4, 5, 6 };

                NdimArray<float> arr3 = arr1 * arr2;

                CollectionAssert.AreEqual(new float[] { 4, 10, 18 }, (float[])arr3);
            }

            {
                NdimArray<double> arr1 = new double[] { 1, 2, 3 };
                NdimArray<double> arr2 = new double[] { 4, 5, 6 };

                NdimArray<double> arr3 = arr1 * arr2;

                CollectionAssert.AreEqual(new double[] { 4, 10, 18 }, (double[])arr3);
            }

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<short> arr1 = new short[] { 1, 2, 3 };
                NdimArray<short> arr2 = new short[] { 4, 5, 6 };

                NdimArray<short> arr3 = arr1 + arr2;
            });
        }

        [TestMethod]
        public void BinaryDivTest() {
            {
                NdimArray<int> arr1 = new int[] { 1, 2, 3 };
                NdimArray<int> arr2 = new int[] { 4, 5, 6 };

                NdimArray<int> arr3 = arr1 / arr2;

                CollectionAssert.AreEqual(new int[] { 1 / 4, 2 / 5, 3 / 6 }, (int[])arr3);
            }

            {
                NdimArray<long> arr1 = new long[] { 1, 2, 3 };
                NdimArray<long> arr2 = new long[] { 4, 5, 6 };

                NdimArray<long> arr3 = arr1 / arr2;

                CollectionAssert.AreEqual(new long[] { 1 / 4, 2 / 5, 3 / 6 }, (long[])arr3);
            }

            {
                NdimArray<float> arr1 = new float[] { 1, 2, 3 };
                NdimArray<float> arr2 = new float[] { 4, 5, 6 };

                NdimArray<float> arr3 = arr1 / arr2;

                CollectionAssert.AreEqual(new float[] { 1f / 4, 2f / 5, 3f / 6 }, (float[])arr3);
            }

            {
                NdimArray<double> arr1 = new double[] { 1, 2, 3 };
                NdimArray<double> arr2 = new double[] { 4, 5, 6 };

                NdimArray<double> arr3 = arr1 / arr2;

                CollectionAssert.AreEqual(new double[] { 1d / 4, 2d / 5, 3d / 6 }, (double[])arr3);
            }

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<short> arr1 = new short[] { 1, 2, 3 };
                NdimArray<short> arr2 = new short[] { 4, 5, 6 };

                NdimArray<short> arr3 = arr1 + arr2;
            });
        }

        [TestMethod]
        public void UnaryPlusTest() {
            {
                NdimArray<int> arr1 = new int[] { 1, 2, 3 };

                NdimArray<int> arr3 = +arr1;

                CollectionAssert.AreEqual(new int[] { 1, 2, 3 }, (int[])arr3);
            }

            {
                NdimArray<long> arr1 = new long[] { 1, 2, 3 };

                NdimArray<long> arr3 = +arr1;

                CollectionAssert.AreEqual(new long[] { 1, 2, 3 }, (long[])arr3);
            }

            {
                NdimArray<float> arr1 = new float[] { 1, 2, 3 };

                NdimArray<float> arr3 = +arr1;

                CollectionAssert.AreEqual(new float[] { 1, 2, 3 }, (float[])arr3);
            }

            {
                NdimArray<double> arr1 = new double[] { 1, 2, 3 };

                NdimArray<double> arr3 = +arr1;

                CollectionAssert.AreEqual(new double[] { 1, 2, 3 }, (double[])arr3);
            }
        }

        [TestMethod]
        public void UnaryMinusTest() {
            {
                NdimArray<int> arr1 = new int[] { 1, 2, 3 };

                NdimArray<int> arr3 = -arr1;

                CollectionAssert.AreEqual(new int[] { -1, -2, -3 }, (int[])arr3);
            }

            {
                NdimArray<long> arr1 = new long[] { 1, 2, 3 };

                NdimArray<long> arr3 = -arr1;

                CollectionAssert.AreEqual(new long[] { -1, -2, -3 }, (long[])arr3);
            }

            {
                NdimArray<float> arr1 = new float[] { 1, 2, 3 };

                NdimArray<float> arr3 = -arr1;

                CollectionAssert.AreEqual(new float[] { -1, -2, -3 }, (float[])arr3);
            }

            {
                NdimArray<double> arr1 = new double[] { 1, 2, 3 };

                NdimArray<double> arr3 = -arr1;

                CollectionAssert.AreEqual(new double[] { -1, -2, -3 }, (double[])arr3);
            }
        }

        [TestMethod]
        public void BinaryScalarAddTest() {
            {
                NdimArray<int> arr1 = new int[] { 1, 2, 3 };

                NdimArray<int> arr3 = arr1 + 2;

                CollectionAssert.AreEqual(new int[] { 3, 4, 5 }, (int[])arr3);
            }

            {
                NdimArray<long> arr1 = new long[] { 1, 2, 3 };

                NdimArray<long> arr3 = arr1 + 2L;

                CollectionAssert.AreEqual(new long[] { 3, 4, 5 }, (long[])arr3);
            }

            {
                NdimArray<float> arr1 = new float[] { 1, 2, 3 };

                NdimArray<float> arr3 = arr1 + 2f;

                CollectionAssert.AreEqual(new float[] { 3, 4, 5 }, (float[])arr3);
            }

            {
                NdimArray<double> arr1 = new double[] { 1, 2, 3 };

                NdimArray<double> arr3 = arr1 + 2d;

                CollectionAssert.AreEqual(new double[] { 3, 4, 5 }, (double[])arr3);
            }

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<short> arr1 = new short[] { 1, 2, 3 };

                NdimArray<short> arr3 = arr1 + (short)2;
            });

            {
                NdimArray<int> arr1 = new int[] { 1, 2, 3 };

                NdimArray<int> arr3 = 2 + arr1;

                CollectionAssert.AreEqual(new int[] { 3, 4, 5 }, (int[])arr3);
            }

            {
                NdimArray<long> arr1 = new long[] { 1, 2, 3 };

                NdimArray<long> arr3 = 2L + arr1;

                CollectionAssert.AreEqual(new long[] { 3, 4, 5 }, (long[])arr3);
            }

            {
                NdimArray<float> arr1 = new float[] { 1, 2, 3 };

                NdimArray<float> arr3 = 2f + arr1;

                CollectionAssert.AreEqual(new float[] { 3, 4, 5 }, (float[])arr3);
            }

            {
                NdimArray<double> arr1 = new double[] { 1, 2, 3 };

                NdimArray<double> arr3 = 2d + arr1;

                CollectionAssert.AreEqual(new double[] { 3, 4, 5 }, (double[])arr3);
            }

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<short> arr1 = new short[] { 1, 2, 3 };

                NdimArray<short> arr3 = (short)2 + arr1;
            });
        }

        [TestMethod]
        public void BinaryScalarSubTest() {
            {
                NdimArray<int> arr1 = new int[] { 1, 2, 3 };

                NdimArray<int> arr3 = arr1 - 2;

                CollectionAssert.AreEqual(new int[] { -1, 0, 1 }, (int[])arr3);
            }

            {
                NdimArray<long> arr1 = new long[] { 1, 2, 3 };

                NdimArray<long> arr3 = arr1 - 2L;

                CollectionAssert.AreEqual(new long[] { -1, 0, 1 }, (long[])arr3);
            }

            {
                NdimArray<float> arr1 = new float[] { 1, 2, 3 };

                NdimArray<float> arr3 = arr1 - 2f;

                CollectionAssert.AreEqual(new float[] { -1, 0, 1 }, (float[])arr3);
            }

            {
                NdimArray<double> arr1 = new double[] { 1, 2, 3 };

                NdimArray<double> arr3 = arr1 - 2d;

                CollectionAssert.AreEqual(new double[] { -1, 0, 1 }, (double[])arr3);
            }

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<short> arr1 = new short[] { 1, 2, 3 };

                NdimArray<short> arr3 = arr1 - (short)2;
            });

            {
                NdimArray<int> arr1 = new int[] { 1, 2, 3 };

                NdimArray<int> arr3 = 2 - arr1;

                CollectionAssert.AreEqual(new int[] { 1, 0, -1 }, (int[])arr3);
            }

            {
                NdimArray<long> arr1 = new long[] { 1, 2, 3 };

                NdimArray<long> arr3 = 2L - arr1;

                CollectionAssert.AreEqual(new long[] { 1, 0, -1 }, (long[])arr3);
            }

            {
                NdimArray<float> arr1 = new float[] { 1, 2, 3 };

                NdimArray<float> arr3 = 2f - arr1;

                CollectionAssert.AreEqual(new float[] { 1, 0, -1 }, (float[])arr3);
            }

            {
                NdimArray<double> arr1 = new double[] { 1, 2, 3 };

                NdimArray<double> arr3 = 2d - arr1;

                CollectionAssert.AreEqual(new double[] { 1, 0, -1 }, (double[])arr3);
            }

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<short> arr1 = new short[] { 1, 2, 3 };

                NdimArray<short> arr3 = (short)2 - arr1;
            });
        }

        [TestMethod]
        public void BinaryScalarMulTest() {
            {
                NdimArray<int> arr1 = new int[] { 1, 2, 3 };

                NdimArray<int> arr3 = arr1 * 2;

                CollectionAssert.AreEqual(new int[] { 2, 4, 6 }, (int[])arr3);
            }

            {
                NdimArray<long> arr1 = new long[] { 1, 2, 3 };

                NdimArray<long> arr3 = arr1 * 2L;

                CollectionAssert.AreEqual(new long[] { 2, 4, 6 }, (long[])arr3);
            }

            {
                NdimArray<float> arr1 = new float[] { 1, 2, 3 };

                NdimArray<float> arr3 = arr1 * 2f;

                CollectionAssert.AreEqual(new float[] { 2, 4, 6 }, (float[])arr3);
            }

            {
                NdimArray<double> arr1 = new double[] { 1, 2, 3 };

                NdimArray<double> arr3 = arr1 * 2d;

                CollectionAssert.AreEqual(new double[] { 2, 4, 6 }, (double[])arr3);
            }

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<short> arr1 = new short[] { 1, 2, 3 };

                NdimArray<short> arr3 = arr1 * (short)2;
            });

            {
                NdimArray<int> arr1 = new int[] { 1, 2, 3 };

                NdimArray<int> arr3 = 2 * arr1;

                CollectionAssert.AreEqual(new int[] { 2, 4, 6 }, (int[])arr3);
            }

            {
                NdimArray<long> arr1 = new long[] { 1, 2, 3 };

                NdimArray<long> arr3 = 2L * arr1;

                CollectionAssert.AreEqual(new long[] { 2, 4, 6 }, (long[])arr3);
            }

            {
                NdimArray<float> arr1 = new float[] { 1, 2, 3 };

                NdimArray<float> arr3 = 2f * arr1;

                CollectionAssert.AreEqual(new float[] { 2, 4, 6 }, (float[])arr3);
            }

            {
                NdimArray<double> arr1 = new double[] { 1, 2, 3 };

                NdimArray<double> arr3 = 2d * arr1;

                CollectionAssert.AreEqual(new double[] { 2, 4, 6 }, (double[])arr3);
            }

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<short> arr1 = new short[] { 1, 2, 3 };

                NdimArray<short> arr3 = (short)2 * arr1;
            });
        }

        [TestMethod]
        public void BinaryScalarDivTest() {
            {
                NdimArray<int> arr1 = new int[] { 1, 2, 3 };

                NdimArray<int> arr3 = arr1 / 2;

                CollectionAssert.AreEqual(new int[] { 1 / 2, 2 / 2, 3 / 2 }, (int[])arr3);
            }

            {
                NdimArray<long> arr1 = new long[] { 1, 2, 3 };

                NdimArray<long> arr3 = arr1 / 2L;

                CollectionAssert.AreEqual(new long[] { 1 / 2, 2 / 2, 3 / 2 }, (long[])arr3);
            }

            {
                NdimArray<float> arr1 = new float[] { 1, 2, 3 };

                NdimArray<float> arr3 = arr1 / 2f;

                CollectionAssert.AreEqual(new float[] { 1f / 2, 2f / 2, 3f / 2 }, (float[])arr3);
            }

            {
                NdimArray<double> arr1 = new double[] { 1, 2, 3 };

                NdimArray<double> arr3 = arr1 / 2d;

                CollectionAssert.AreEqual(new double[] { 1d / 2, 2d / 2, 3d / 2 }, (double[])arr3);
            }

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<short> arr1 = new short[] { 1, 2, 3 };

                NdimArray<short> arr3 = arr1 / (short)2;
            });

            {
                NdimArray<int> arr1 = new int[] { 1, 2, 3 };

                NdimArray<int> arr3 = 2 / arr1;

                CollectionAssert.AreEqual(new int[] { 2 / 1, 2 / 2, 2 / 3 }, (int[])arr3);
            }

            {
                NdimArray<long> arr1 = new long[] { 1, 2, 3 };

                NdimArray<long> arr3 = 2L / arr1;

                CollectionAssert.AreEqual(new long[] { 2 / 1, 2 / 2, 2 / 3 }, (long[])arr3);
            }

            {
                NdimArray<float> arr1 = new float[] { 1, 2, 3 };

                NdimArray<float> arr3 = 2f / arr1;

                CollectionAssert.AreEqual(new float[] { 2f / 1, 2f / 2, 2f / 3 }, (float[])arr3);
            }

            {
                NdimArray<double> arr1 = new double[] { 1, 2, 3 };

                NdimArray<double> arr3 = 2d / arr1;

                CollectionAssert.AreEqual(new double[] { 2d / 1, 2d / 2, 2d / 3 }, (double[])arr3);
            }

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<short> arr1 = new short[] { 1, 2, 3 };

                NdimArray<short> arr3 = (short)2 / arr1;
            });
        }
    }
}
