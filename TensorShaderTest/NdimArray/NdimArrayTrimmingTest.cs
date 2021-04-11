using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest.NdimArray {
    [TestClass]
    public class NdimArrayTrimmingTest {
        [TestMethod]
        public void TrimmingTest1() {
            Shape shape = new(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 });
            (int ta, int tb)[][] trims_list = new (int ta, int tb)[][]{
                new (int ta, int tb)[] { (1, 2), (3, 4), (2, 2), (4, 3), (2, 1) },
                new (int ta, int tb)[] { (0, 0), (3, 4), (2, 2), (4, 3), (2, 1) },
                new (int ta, int tb)[] { (1, 2), (3, 4), (2, 2), (4, 3), (0, 0) },
                new (int ta, int tb)[] { (1, 2), (3, 4), (0, 0), (4, 3), (2, 1) },
                new (int ta, int tb)[] { (0, 1), (3, 4), (2, 2), (4, 3), (2, 1) },
                new (int ta, int tb)[] { (1, 2), (3, 4), (2, 2), (4, 3), (0, 1) },
                new (int ta, int tb)[] { (1, 2), (3, 4), (0, 1), (4, 3), (2, 1) },
                new (int ta, int tb)[] { (1, 0), (3, 4), (2, 2), (4, 3), (2, 1) },
                new (int ta, int tb)[] { (1, 2), (3, 4), (2, 2), (4, 3), (1, 0) },
                new (int ta, int tb)[] { (1, 2), (3, 4), (1, 0), (4, 3), (2, 1) },
            };

            foreach (var trims in trims_list) {
                Random random = new();

                float[] v = (new float[shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();

                NdimArray<float> arr = new(shape, v);
                NdimArray<float> new_arr = NdimArray<float>.Trimming(arr, trims);

                CheckTrimming(trims, arr, new_arr);
            }
        }

        [TestMethod]
        public void TrimmingTest2() {
            (Shape shape, (int ta, int tb)[] trims)[] tests = new (Shape shape, (int ta, int tb)[] trims)[]{
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7 }), new (int ta, int tb)[] { (1, 1) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8 }), new (int ta, int tb)[] { (1, 1), (1, 1) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 }), new (int ta, int tb)[] { (1, 1), (1, 1), (1, 1) }),
            };

            foreach ((Shape shape, (int ta, int tb)[] trims) in tests) {
                Random random = new();

                float[] v = (new float[shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();

                NdimArray<float> arr = new(shape, v);
                NdimArray<float> new_arr = NdimArray<float>.TrimmingND(arr, trims);

                (int, int)[] new_trims = (new (int, int)[] { (0, 0) }).Concat(trims).Concat(new (int, int)[] { (0, 0) }).ToArray();

                CheckTrimming(new_trims, arr, new_arr);
            }
        }

        [TestMethod]
        public void SliceTest() {
            (Shape shape, (int offset, int count)[] range, (int ta, int tb)[] trims)[] tests =
                new (Shape shape, (int offset, int count)[] range, (int ta, int tb)[] trims)[]{

                (new Shape(ShapeType.Map, new int[] { 5, 9, 7 }), new (int offset, int count)[] { (1, 6) }, new (int ta, int tb)[] { (1, 2) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8 }), new (int offset, int count)[] { (1, 6), (2, 5) }, new (int ta, int tb)[] { (1, 2), (2, 0) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 }), new (int offset, int count)[] { (1, 6), (2, 5), (3, 4) }, new (int ta, int tb)[] { (1, 2), (2, 0), (3, 1) }),
            };

            foreach ((Shape shape, (int offset, int count)[] range, (int ta, int tb)[] trims) in tests) {
                Random random = new();

                float[] v = (new float[shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();

                NdimArray<float> arr = new(shape, v);
                NdimArray<float> new_arr = NdimArray<float>.SliceND(arr, range);

                (int, int)[] new_trims = (new (int, int)[] { (0, 0) }).Concat(range).Concat(new (int, int)[] { (0, 0) }).ToArray();

                CheckTrimming(new_trims, arr, new_arr);
            }
        }

        private static void CheckTrimming((int ta, int tb)[] trims, NdimArray<float> arr, NdimArray<float> new_arr) {
            Shape new_shape = new_arr.Shape;

            if (arr.Ndim == 3) {
                for (int i2 = 0; i2 < new_shape[2]; i2++) {
                    for (int i1 = 0; i1 < new_shape[1]; i1++) {
                        for (int i0 = 0; i0 < new_shape[0]; i0++) {
                            Assert.AreEqual(
                                arr[i0 + trims[0].ta, i1 + trims[1].ta, i2 + trims[2].ta],
                                new_arr[i0, i1, i2]
                            );
                        }
                    }
                }

                return;
            }

            if (arr.Ndim == 4) {
                for (int i3 = 0; i3 < new_shape[3]; i3++) {
                    for (int i2 = 0; i2 < new_shape[2]; i2++) {
                        for (int i1 = 0; i1 < new_shape[1]; i1++) {
                            for (int i0 = 0; i0 < new_shape[0]; i0++) {
                                Assert.AreEqual(
                                    arr[i0 + trims[0].ta, i1 + trims[1].ta, i2 + trims[2].ta, i3 + trims[3].ta],
                                    new_arr[i0, i1, i2, i3]
                                );
                            }
                        }
                    }
                }

                return;
            }

            if (arr.Ndim == 5) {
                for (int i4 = 0; i4 < new_shape[4]; i4++) {
                    for (int i3 = 0; i3 < new_shape[3]; i3++) {
                        for (int i2 = 0; i2 < new_shape[2]; i2++) {
                            for (int i1 = 0; i1 < new_shape[1]; i1++) {
                                for (int i0 = 0; i0 < new_shape[0]; i0++) {
                                    Assert.AreEqual(
                                        arr[i0 + trims[0].ta, i1 + trims[1].ta, i2 + trims[2].ta, i3 + trims[3].ta, i4 + trims[4].ta],
                                        new_arr[i0, i1, i2, i3, i4]
                                    );
                                }
                            }
                        }
                    }
                }

                return;
            }

            Assert.Fail();
        }
    }
}
