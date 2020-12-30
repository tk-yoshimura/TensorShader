using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest {
    [TestClass]
    public class NdimArrayPaddingTest {
        [TestMethod]
        public void ZeroPaddingTest() {
            Random random = new Random();

            Shape shape = new Shape(ShapeType.Map, new int[] { 3, 4, 5, 6, 7 });
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
                float[] v = (new float[shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();

                NdimArray<float> arr = new NdimArray<float>(shape, v);
                NdimArray<float> new_arr = NdimArray<float>.ZeroPadding(arr, pads);

                Shape new_shape = new_arr.Shape;

                for (int i4 = 0; i4 < new_shape[4]; i4++) {
                    for (int i3 = 0; i3 < new_shape[3]; i3++) {
                        for (int i2 = 0; i2 < new_shape[2]; i2++) {
                            for (int i1 = 0; i1 < new_shape[1]; i1++) {
                                for (int i0 = 0; i0 < new_shape[0]; i0++) {

                                    if (i4 < pads[4].pa || i4 >= pads[4].pa + shape[4]) {
                                        Assert.AreEqual(0f, new_arr[i4, i3, i2, i1, i0]);
                                        continue;
                                    }
                                    if (i3 < pads[3].pa || i3 >= pads[3].pa + shape[3]) {
                                        Assert.AreEqual(0f, new_arr[i4, i3, i2, i1, i0]);
                                        continue;
                                    }
                                    if (i2 < pads[2].pa || i2 >= pads[2].pa + shape[2]) {
                                        Assert.AreEqual(0f, new_arr[i4, i3, i2, i1, i0]);
                                        continue;
                                    }
                                    if (i1 < pads[1].pa || i1 >= pads[1].pa + shape[1]) {
                                        Assert.AreEqual(0f, new_arr[i4, i3, i2, i1, i0]);
                                        continue;
                                    }
                                    if (i0 < pads[0].pa || i0 >= pads[0].pa + shape[0]) {
                                        Assert.AreEqual(0f, new_arr[i4, i3, i2, i1, i0]);
                                        continue;
                                    }

                                    Assert.AreEqual(
                                        arr[i4 - pads[4].pa, i3 - pads[3].pa, i2 - pads[2].pa, i1 - pads[1].pa, i0 - pads[0].pa], 
                                        new_arr[i4, i3, i2, i1, i0]
                                    );

                                }
                            }
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void EdgePaddingTest() {
            Random random = new Random();

            Shape shape = new Shape(ShapeType.Map, new int[] { 3, 4, 5, 6, 7 });
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
                float[] v = (new float[shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();

                NdimArray<float> arr = new NdimArray<float>(shape, v);
                NdimArray<float> new_arr = NdimArray<float>.EdgePadding(arr, pads);

                Shape new_shape = new_arr.Shape;

                for (int i4 = 0; i4 < new_shape[4]; i4++) {
                    for (int i3 = 0; i3 < new_shape[3]; i3++) {
                        for (int i2 = 0; i2 < new_shape[2]; i2++) {
                            for (int i1 = 0; i1 < new_shape[1]; i1++) {
                                for (int i0 = 0; i0 < new_shape[0]; i0++) {

                                    Assert.AreEqual(
                                        arr[
                                            Math.Min(Math.Max(0, i4 - pads[4].pa), shape[4] - 1), 
                                            Math.Min(Math.Max(0, i3 - pads[3].pa), shape[3] - 1), 
                                            Math.Min(Math.Max(0, i2 - pads[2].pa), shape[2] - 1), 
                                            Math.Min(Math.Max(0, i1 - pads[1].pa), shape[1] - 1), 
                                            Math.Min(Math.Max(0, i0 - pads[0].pa), shape[0] - 1)
                                        ], 
                                        new_arr[i4, i3, i2, i1, i0]
                                    );

                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
