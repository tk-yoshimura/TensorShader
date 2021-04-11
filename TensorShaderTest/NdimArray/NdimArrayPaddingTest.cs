using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest.NdimArray {
    [TestClass]
    public class NdimArrayPaddingTest {
        [TestMethod]
        public void ZeroPaddingTest1() {
            Shape shape = new(ShapeType.Map, new int[] { 3, 4, 5, 6, 7 });
            (int pa, int pb)[][] pads_list = new (int pa, int pb)[][]{
                new (int pa, int pb)[] { (1, 2), (3, 4), (2, 2), (4, 3), (2, 1) },
                new (int pa, int pb)[] { (0, 0), (3, 4), (2, 2), (4, 3), (2, 1) },
                new (int pa, int pb)[] { (1, 2), (3, 4), (2, 2), (4, 3), (0, 0) },
                new (int pa, int pb)[] { (1, 2), (3, 4), (0, 0), (4, 3), (2, 1) },
                new (int pa, int pb)[] { (0, 1), (3, 4), (2, 2), (4, 3), (2, 1) },
                new (int pa, int pb)[] { (1, 2), (3, 4), (2, 2), (4, 3), (0, 1) },
                new (int pa, int pb)[] { (1, 2), (3, 4), (0, 1), (4, 3), (2, 1) },
                new (int pa, int pb)[] { (1, 0), (3, 4), (2, 2), (4, 3), (2, 1) },
                new (int pa, int pb)[] { (1, 2), (3, 4), (2, 2), (4, 3), (1, 0) },
                new (int pa, int pb)[] { (1, 2), (3, 4), (1, 0), (4, 3), (2, 1) },
            };

            foreach (var pads in pads_list) {
                Random random = new();
                float[] v = (new float[shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();

                NdimArray<float> arr = new(shape, v);
                NdimArray<float> new_arr = NdimArray<float>.ZeroPadding(arr, pads);

                CheckZeroPadding(shape, pads, arr, new_arr);
            }
        }

        [TestMethod]
        public void ZeroPaddingTest2() {
            (Shape shape, (int pa, int pb)[] pads)[] tests = new (Shape shape, (int pa, int pb)[] pads)[]{
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7 }), new (int pa, int pb)[] { (1, 1) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8 }), new (int pa, int pb)[] { (1, 1), (1, 1) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 }), new (int pa, int pb)[] { (1, 1), (1, 1), (1, 1) }),
            };

            foreach ((Shape shape, (int pa, int pb)[] pads) in tests) {
                Random random = new();

                float[] v = (new float[shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();

                NdimArray<float> arr = new(shape, v);
                NdimArray<float> new_arr = NdimArray<float>.ZeroPaddingND(arr, pads);

                (int, int)[] new_pads = (new (int, int)[] { (0, 0) }).Concat(pads).Concat(new (int, int)[] { (0, 0) }).ToArray();

                CheckZeroPadding(shape, new_pads, arr, new_arr);
            }
        }

        [TestMethod]
        public void EdgePaddingTest1() {
            Shape shape = new(ShapeType.Map, new int[] { 3, 4, 5, 6, 7 });
            (int pa, int pb)[][] pads_list = new (int pa, int pb)[][]{
                new (int pa, int pb)[] { (1, 2), (3, 4), (2, 2), (4, 3), (2, 1) },
                new (int pa, int pb)[] { (0, 0), (3, 4), (2, 2), (4, 3), (2, 1) },
                new (int pa, int pb)[] { (1, 2), (3, 4), (2, 2), (4, 3), (0, 0) },
                new (int pa, int pb)[] { (1, 2), (3, 4), (0, 0), (4, 3), (2, 1) },
                new (int pa, int pb)[] { (0, 1), (3, 4), (2, 2), (4, 3), (2, 1) },
                new (int pa, int pb)[] { (1, 2), (3, 4), (2, 2), (4, 3), (0, 1) },
                new (int pa, int pb)[] { (1, 2), (3, 4), (0, 1), (4, 3), (2, 1) },
                new (int pa, int pb)[] { (1, 0), (3, 4), (2, 2), (4, 3), (2, 1) },
                new (int pa, int pb)[] { (1, 2), (3, 4), (2, 2), (4, 3), (1, 0) },
                new (int pa, int pb)[] { (1, 2), (3, 4), (1, 0), (4, 3), (2, 1) },
            };

            foreach (var pads in pads_list) {
                Random random = new();
                float[] v = (new float[shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();

                NdimArray<float> arr = new(shape, v);
                NdimArray<float> new_arr = NdimArray<float>.EdgePadding(arr, pads);

                CheckEdgePadding(shape, pads, arr, new_arr);
            }
        }

        [TestMethod]
        public void EdgePaddingTest2() {
            (Shape shape, (int pa, int pb)[] pads)[] tests = new (Shape shape, (int pa, int pb)[] pads)[]{
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7 }), new (int pa, int pb)[] { (1, 1) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8 }), new (int pa, int pb)[] { (1, 1), (1, 1) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 }), new (int pa, int pb)[] { (1, 1), (1, 1), (1, 1) }),
            };

            foreach ((Shape shape, (int pa, int pb)[] pads) in tests) {
                Random random = new();

                float[] v = (new float[shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();

                NdimArray<float> arr = new(shape, v);
                NdimArray<float> new_arr = NdimArray<float>.EdgePaddingND(arr, pads);

                (int, int)[] new_pads = (new (int, int)[] { (0, 0) }).Concat(pads).Concat(new (int, int)[] { (0, 0) }).ToArray();

                CheckEdgePadding(shape, new_pads, arr, new_arr);
            }
        }

        private static void CheckZeroPadding(Shape shape, (int pa, int pb)[] pads, NdimArray<float> arr, NdimArray<float> new_arr) {
            Shape new_shape = new_arr.Shape;

            if (arr.Ndim == 3) {
                for (int i2 = 0; i2 < new_shape[2]; i2++) {
                    for (int i1 = 0; i1 < new_shape[1]; i1++) {
                        for (int i0 = 0; i0 < new_shape[0]; i0++) {

                            if (i2 < pads[2].pa || i2 >= pads[2].pa + shape[2] ||
                                i1 < pads[1].pa || i1 >= pads[1].pa + shape[1] ||
                                i0 < pads[0].pa || i0 >= pads[0].pa + shape[0]) {

                                Assert.AreEqual(0f, new_arr[i0, i1, i2]);
                                continue;
                            }

                            Assert.AreEqual(
                                arr[i0 - pads[0].pa, i1 - pads[1].pa, i2 - pads[2].pa],
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

                                if (i3 < pads[3].pa || i3 >= pads[3].pa + shape[3] ||
                                    i2 < pads[2].pa || i2 >= pads[2].pa + shape[2] ||
                                    i1 < pads[1].pa || i1 >= pads[1].pa + shape[1] ||
                                    i0 < pads[0].pa || i0 >= pads[0].pa + shape[0]) {

                                    Assert.AreEqual(0f, new_arr[i0, i1, i2, i3]);
                                    continue;
                                }

                                Assert.AreEqual(
                                    arr[i0 - pads[0].pa, i1 - pads[1].pa, i2 - pads[2].pa, i3 - pads[3].pa],
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

                                    if (i4 < pads[4].pa || i4 >= pads[4].pa + shape[4] ||
                                        i3 < pads[3].pa || i3 >= pads[3].pa + shape[3] ||
                                        i2 < pads[2].pa || i2 >= pads[2].pa + shape[2] ||
                                        i1 < pads[1].pa || i1 >= pads[1].pa + shape[1] ||
                                        i0 < pads[0].pa || i0 >= pads[0].pa + shape[0]) {

                                        Assert.AreEqual(0f, new_arr[i0, i1, i2, i3, i4]);
                                        continue;
                                    }

                                    Assert.AreEqual(
                                        arr[i0 - pads[0].pa, i1 - pads[1].pa, i2 - pads[2].pa, i3 - pads[3].pa, i4 - pads[4].pa],
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

        private static void CheckEdgePadding(Shape shape, (int pa, int pb)[] pads, NdimArray<float> arr, NdimArray<float> new_arr) {
            Shape new_shape = new_arr.Shape;

            if (arr.Ndim == 3) {
                for (int i2 = 0; i2 < new_shape[2]; i2++) {
                    for (int i1 = 0; i1 < new_shape[1]; i1++) {
                        for (int i0 = 0; i0 < new_shape[0]; i0++) {

                            Assert.AreEqual(
                                arr[
                                    Math.Min(Math.Max(0, i0 - pads[0].pa), shape[0] - 1),
                                    Math.Min(Math.Max(0, i1 - pads[1].pa), shape[1] - 1),
                                    Math.Min(Math.Max(0, i2 - pads[2].pa), shape[2] - 1)
                                ],
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
                                    arr[
                                        Math.Min(Math.Max(0, i0 - pads[0].pa), shape[0] - 1),
                                        Math.Min(Math.Max(0, i1 - pads[1].pa), shape[1] - 1),
                                        Math.Min(Math.Max(0, i2 - pads[2].pa), shape[2] - 1),
                                        Math.Min(Math.Max(0, i3 - pads[3].pa), shape[3] - 1)
                                    ],
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
                                        arr[
                                            Math.Min(Math.Max(0, i0 - pads[0].pa), shape[0] - 1),
                                            Math.Min(Math.Max(0, i1 - pads[1].pa), shape[1] - 1),
                                            Math.Min(Math.Max(0, i2 - pads[2].pa), shape[2] - 1),
                                            Math.Min(Math.Max(0, i3 - pads[3].pa), shape[3] - 1),
                                            Math.Min(Math.Max(0, i4 - pads[4].pa), shape[4] - 1)
                                        ],
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
