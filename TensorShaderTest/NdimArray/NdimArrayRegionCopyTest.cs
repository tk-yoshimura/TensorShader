﻿using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest.NdimArray {
    [TestClass]
    public class NdimArrayRegionCopyTest {
        [TestMethod]
        public void RegionCopyTest1() {
            Shape src_shape = new Shape(ShapeType.Map, new int[] { 3, 4, 3, 5, 6 });
            Shape dst_shape = new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 });
            int[][] ps_list = new int[][]{
                new int[] { 1, 0, 0, 0, 0 },
                new int[] { 0, 1, 0, 0, 0 },
                new int[] { 0, 0, 1, 0, 0 },
                new int[] { 0, 0, 0, 1, 0 },
                new int[] { 0, 0, 0, 0, 1 },
                new int[] { 1, 1, 0, 0, 0 },
                new int[] { 0, 2, 2, 0, 0 },
                new int[] { 0, 0, 3, 3, 0 },
                new int[] { 1, 2, 3, 3, 3 }
            };

            foreach (var ps in ps_list) {
                Random random = new Random();

                float[] src_v = (new float[src_shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();
                float[] dst_v = (new float[dst_shape.Length]).Select((v) => -(float)random.NextDouble()).ToArray();

                NdimArray<float> src_arr = new NdimArray<float>(src_shape, src_v);
                NdimArray<float> dst_arr = new NdimArray<float>(dst_shape, dst_v);
                NdimArray<float> ret_arr = dst_arr.Copy();

                NdimArray<float>.RegionCopy(src_arr, ret_arr, ps);

                CheckRegionCopy(src_shape, dst_shape, ps, src_arr, dst_arr, ret_arr);
            }
        }

        [TestMethod]
        public void RegionCopyTest2() {
            Shape src_shape = new Shape(ShapeType.Map, new int[] { 3, 4, 3, 5, 6 });
            (Shape dst_shape, int[] ps)[] tests = new (Shape dst_shape, int[] ps)[] {
                (new Shape(ShapeType.Map, new int[] { 3, 5, 4, 6, 7 }), new int[] { 0, 1, 0, 0, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 5, 4, 6, 7 }), new int[] { 0, 0, 1, 0, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 5, 4, 6, 7 }), new int[] { 0, 0, 0, 1, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 5, 4, 6, 7 }), new int[] { 0, 0, 0, 0, 1 }),
                (new Shape(ShapeType.Map, new int[] { 3, 5, 4, 6, 7 }), new int[] { 0, 1, 1, 0, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 5, 4, 6, 7 }), new int[] { 0, 0, 1, 1, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 5, 4, 6, 7 }), new int[] { 0, 0, 0, 1, 1 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 4, 6, 7 }), new int[] { 0, 0, 1, 0, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 4, 6, 7 }), new int[] { 0, 0, 0, 1, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 4, 6, 7 }), new int[] { 0, 0, 0, 0, 1 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 4, 6, 7 }), new int[] { 0, 0, 1, 1, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 4, 6, 7 }), new int[] { 0, 0, 0, 1, 1 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 3, 6, 7 }), new int[] { 0, 0, 0, 1, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 3, 6, 7 }), new int[] { 0, 0, 0, 0, 1 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 3, 6, 7 }), new int[] { 0, 0, 0, 1, 1 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 3, 5, 7 }), new int[] { 0, 0, 0, 0, 1 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 3, 5, 6 }), new int[] { 0, 0, 0, 0, 0 }),
            };

            foreach ((Shape dst_shape, int[] ps) in tests) {
                Random random = new Random();

                float[] src_v = (new float[src_shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();
                float[] dst_v = (new float[dst_shape.Length]).Select((v) => -(float)random.NextDouble()).ToArray();

                NdimArray<float> src_arr = new NdimArray<float>(src_shape, src_v);
                NdimArray<float> dst_arr = new NdimArray<float>(dst_shape, dst_v);
                NdimArray<float> ret_arr = dst_arr.Copy();

                NdimArray<float>.RegionCopy(src_arr, ret_arr, ps);

                CheckRegionCopy(src_shape, dst_shape, ps, src_arr, dst_arr, ret_arr);
            }
        }

        [TestMethod]
        public void BadRegionCopyTest() {
            Shape src_shape = new Shape(ShapeType.Map, new int[] { 3, 4, 3, 5, 6 });
            (Shape dst_shape, int[] ps)[] tests = new (Shape dst_shape, int[] ps)[] {
                (new Shape(ShapeType.Map, new int[] { 3, 5, 4, 6, 7 }), new int[] { 0, 2, 0, 0, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 5, 4, 6, 7 }), new int[] { 0, 0, 2, 0, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 5, 4, 6, 7 }), new int[] { 0, 0, 0, 2, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 5, 4, 6, 7 }), new int[] { 0, 0, 0, 0, 2 }),
                (new Shape(ShapeType.Map, new int[] { 3, 5, 4, 6, 7 }), new int[] { 0, 2, 2, 0, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 5, 4, 6, 7 }), new int[] { 0, 0, 2, 2, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 5, 4, 6, 7 }), new int[] { 0, 0, 0, 2, 2 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 4, 6, 7 }), new int[] { 0, 0, 2, 0, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 4, 6, 7 }), new int[] { 0, 0, 0, 2, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 4, 6, 7 }), new int[] { 0, 0, 0, 0, 2 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 4, 6, 7 }), new int[] { 0, 0, 2, 2, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 4, 6, 7 }), new int[] { 0, 0, 0, 2, 2 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 3, 6, 7 }), new int[] { 0, 0, 0, 2, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 3, 6, 7 }), new int[] { 0, 0, 0, 0, 2 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 3, 6, 7 }), new int[] { 0, 0, 0, 2, 2 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 3, 5, 7 }), new int[] { 0, 0, 0, 0, 2 }),
                (new Shape(ShapeType.Kernel, new int[] { 3, 4, 3, 5, 6 }), new int[] { 0, 0, 0, 0, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 3, 5 }), new int[] { 0, 0, 0, 0, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 3, 5, 6, 7}), new int[] { 0, 0, 0, 0, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 3, 5, 5 }), new int[] { 0, 0, 0, 0, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 3, 4, 5 }), new int[] { 0, 0, 0, 0, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 3, 5, 6 }), new int[] { 0, 0, 0, 0 }),
                (new Shape(ShapeType.Map, new int[] { 3, 4, 3, 5, 6 }), new int[] { 0, 0, 0, 0, 0, 0 }),
            };

            foreach ((Shape dst_shape, int[] ps) in tests) {
                Assert.ThrowsException<ArgumentException>(
                    () => { 
                        Random random = new Random();

                        float[] src_v = (new float[src_shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();
                        float[] dst_v = (new float[dst_shape.Length]).Select((v) => -(float)random.NextDouble()).ToArray();

                        NdimArray<float> src_arr = new NdimArray<float>(src_shape, src_v);
                        NdimArray<float> dst_arr = new NdimArray<float>(dst_shape, dst_v);
                        NdimArray<float> ret_arr = dst_arr.Copy();

                        NdimArray<float>.RegionCopy(src_arr, ret_arr, ps);

                        CheckRegionCopy(src_shape, dst_shape, ps, src_arr, dst_arr, ret_arr);
                    }
                , $"{dst_shape}, {string.Join(',', ps)}");
            }
        }

        private static void CheckRegionCopy(Shape src_shape, Shape dst_shape, int[] ps, NdimArray<float> src_arr, NdimArray<float> dst_arr, NdimArray<float> ret_arr) {
            for (int i4 = 0; i4 < dst_shape[4]; i4++) {
                for (int i3 = 0; i3 < dst_shape[3]; i3++) {
                    for (int i2 = 0; i2 < dst_shape[2]; i2++) {
                        for (int i1 = 0; i1 < dst_shape[1]; i1++) {
                            for (int i0 = 0; i0 < dst_shape[0]; i0++) {

                                if (i0 >= ps[0] && i0 < ps[0] + src_shape[0] &&
                                    i1 >= ps[1] && i1 < ps[1] + src_shape[1] &&
                                    i2 >= ps[2] && i2 < ps[2] + src_shape[2] &&
                                    i3 >= ps[3] && i3 < ps[3] + src_shape[3] &&
                                    i4 >= ps[4] && i4 < ps[4] + src_shape[4]) {

                                    Assert.AreEqual(
                                        src_arr[i0 - ps[0], i1 - ps[1], i2 - ps[2], i3 - ps[3], i4 - ps[4]],
                                        ret_arr[i0, i1, i2, i3, i4]
                                    );
                                }
                                else {
                                    Assert.AreEqual(
                                        dst_arr[i0, i1, i2, i3, i4],
                                        ret_arr[i0, i1, i2, i3, i4]
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