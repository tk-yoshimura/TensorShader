﻿using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest.NdimArray {
    [TestClass]
    public class NdimArrayRegionCopyTest {
        [TestMethod]
        public void RegionCopyTest1() {
            Shape src_shape = new(ShapeType.Map, new int[] { 3, 4, 3, 5, 6 });
            Shape dst_shape = new(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 });
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
                Random random = new();

                float[] src_v = (new float[src_shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();
                float[] dst_v = (new float[dst_shape.Length]).Select((v) => -(float)random.NextDouble()).ToArray();

                NdimArray<float> src_arr = new(src_shape, src_v);
                NdimArray<float> dst_arr = new(dst_shape, dst_v);
                NdimArray<float> ret_arr = dst_arr.Copy();

                NdimArray<float>.RegionCopy(src_arr, ret_arr, ps);

                CheckRegionCopy(src_shape, dst_shape, ps, src_arr, dst_arr, ret_arr);
            }
        }

        [TestMethod]
        public void RegionCopyTest2() {
            Shape src_shape = new(ShapeType.Map, new int[] { 3, 4, 3, 5, 6 });
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
                Random random = new();

                float[] src_v = (new float[src_shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();
                float[] dst_v = (new float[dst_shape.Length]).Select((v) => -(float)random.NextDouble()).ToArray();

                NdimArray<float> src_arr = new(src_shape, src_v);
                NdimArray<float> dst_arr = new(dst_shape, dst_v);
                NdimArray<float> ret_arr = dst_arr.Copy();

                NdimArray<float>.RegionCopy(src_arr, ret_arr, ps);

                CheckRegionCopy(src_shape, dst_shape, ps, src_arr, dst_arr, ret_arr);
            }
        }

        [TestMethod]
        public void RegionCopyTest3() {
            (Shape src_shape, Shape dst_shape, int[] ps)[] tests = new (Shape src_shape, Shape dst_shape, int[] ps)[] {
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7 }), new Shape(ShapeType.Map, new int[] { 5, 10, 7 }), new int[]{ 0 }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7 }), new Shape(ShapeType.Map, new int[] { 5, 10, 7 }), new int[]{ 1 }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 8 }), new int[]{ 0, 0 }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 8 }), new int[]{ 1, 0 }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 8 }), new int[]{ 0, 1 }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 8 }), new int[]{ 1, 1 }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 9, 9 }), new int[]{ 0, 0, 0 }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 9, 9 }), new int[]{ 1, 0, 0 }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 9, 9 }), new int[]{ 0, 1, 0 }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 9, 9 }), new int[]{ 0, 0, 1 }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 9, 9 }), new int[]{ 1, 1, 1 }),
            };

            foreach ((Shape src_shape, Shape dst_shape, int[] ps) in tests) {
                Random random = new();

                float[] src_v = (new float[src_shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();
                float[] dst_v = (new float[dst_shape.Length]).Select((v) => -(float)random.NextDouble()).ToArray();

                NdimArray<float> src_arr = new(src_shape, src_v);
                NdimArray<float> dst_arr = new(dst_shape, dst_v);
                NdimArray<float> ret_arr = dst_arr.Copy();

                NdimArray<float>.RegionCopyND(src_arr, ret_arr, ps);

                int[] new_ps = (new int[] { 0 }).Concat(ps).Concat(new int[] { 0 }).ToArray();

                CheckRegionCopy(src_shape, dst_shape, new_ps, src_arr, dst_arr, ret_arr);
            }
        }

        [TestMethod]
        public void RegionCopyWithSliceTest() {
            (Shape src_shape, Shape dst_shape, (int src_offset, int dst_offset, int count)[] region)[] tests
                = new (Shape src_shape, Shape dst_shape, (int src_offset, int dst_offset, int count)[] region)[] {

                (new Shape(ShapeType.Map, new int[] { 5, 9, 7 }), new Shape(ShapeType.Map, new int[] { 5, 10, 7 }), new (int, int, int)[]{ (0, 1, 9) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7 }), new Shape(ShapeType.Map, new int[] { 5, 10, 7 }), new (int, int, int)[]{ (1, 2, 7) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 8 }), new (int, int, int)[]{ (0, 1, 9), (0, 1, 7) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 8 }), new (int, int, int)[]{ (1, 2, 7), (0, 1, 7) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 8 }), new (int, int, int)[]{ (0, 1, 9), (1, 2, 5) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 8 }), new (int, int, int)[]{ (1, 2, 7), (1, 2, 5) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 9, 9 }), new (int, int, int)[]{ (0, 1, 9), (0, 1, 7), (0, 1, 8) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 9, 9 }), new (int, int, int)[]{ (1, 2, 7), (0, 1, 7), (0, 1, 8) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 9, 9 }), new (int, int, int)[]{ (0, 1, 9), (1, 2, 5), (0, 1, 8) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 9, 9 }), new (int, int, int)[]{ (1, 2, 7), (1, 2, 5), (0, 1, 8) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 9, 9 }), new (int, int, int)[]{ (0, 1, 9), (0, 1, 7), (1, 2, 6) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 9, 9 }), new (int, int, int)[]{ (1, 2, 7), (0, 1, 7), (1, 2, 6) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 9, 9 }), new (int, int, int)[]{ (0, 1, 9), (1, 2, 5), (1, 2, 6) }),
                (new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 }), new Shape(ShapeType.Map, new int[] { 5, 10, 8, 9, 9 }), new (int, int, int)[]{ (1, 2, 7), (1, 2, 5), (1, 2, 6) }),
            };

            foreach ((Shape src_shape, Shape dst_shape, (int src_offset, int dst_offset, int count)[] region) in tests) {
                Random random = new();

                float[] src_v = (new float[src_shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();
                float[] dst_v = (new float[dst_shape.Length]).Select((v) => -(float)random.NextDouble()).ToArray();

                NdimArray<float> src_arr = new(src_shape, src_v);
                NdimArray<float> dst_arr = new(dst_shape, dst_v);
                NdimArray<float> ret_arr = dst_arr.Copy();

                NdimArray<float>.RegionCopyND(src_arr, ret_arr, region);

                (int, int, int)[] new_region = (new (int, int, int)[] { (0, 0, src_arr.Channels) }).Concat(region).Concat(new (int, int, int)[] { (0, 0, src_arr.Batch) }).ToArray();

                CheckRegionCopy(src_shape, dst_shape, new_region, src_arr, dst_arr, ret_arr);
            }
        }

        [TestMethod]
        public void BadRegionCopyTest() {
            Shape src_shape = new(ShapeType.Map, new int[] { 3, 4, 3, 5, 6 });
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
                        Random random = new();

                        float[] src_v = (new float[src_shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();
                        float[] dst_v = (new float[dst_shape.Length]).Select((v) => -(float)random.NextDouble()).ToArray();

                        NdimArray<float> src_arr = new(src_shape, src_v);
                        NdimArray<float> dst_arr = new(dst_shape, dst_v);
                        NdimArray<float> ret_arr = dst_arr.Copy();

                        NdimArray<float>.RegionCopy(src_arr, ret_arr, ps);

                        CheckRegionCopy(src_shape, dst_shape, ps, src_arr, dst_arr, ret_arr);
                    }
                , $"{dst_shape}, {string.Join(',', ps)}");
            }
        }

        private static void CheckRegionCopy(Shape src_shape, Shape dst_shape, int[] ps, NdimArray<float> src_arr, NdimArray<float> dst_arr, NdimArray<float> ret_arr) {

            if (src_shape.Ndim == 3) {
                for (int i2 = 0; i2 < dst_shape[2]; i2++) {
                    for (int i1 = 0; i1 < dst_shape[1]; i1++) {
                        for (int i0 = 0; i0 < dst_shape[0]; i0++) {

                            if (i0 >= ps[0] && i0 < ps[0] + src_shape[0] &&
                                i1 >= ps[1] && i1 < ps[1] + src_shape[1] &&
                                i2 >= ps[2] && i2 < ps[2] + src_shape[2]) {

                                Assert.AreEqual(
                                    src_arr[i0 - ps[0], i1 - ps[1], i2 - ps[2]],
                                    ret_arr[i0, i1, i2]
                                );
                            }
                            else {
                                Assert.AreEqual(
                                    dst_arr[i0, i1, i2],
                                    ret_arr[i0, i1, i2]
                                );
                            }
                        }
                    }
                }

                return;
            }

            if (src_shape.Ndim == 4) {
                for (int i3 = 0; i3 < dst_shape[3]; i3++) {
                    for (int i2 = 0; i2 < dst_shape[2]; i2++) {
                        for (int i1 = 0; i1 < dst_shape[1]; i1++) {
                            for (int i0 = 0; i0 < dst_shape[0]; i0++) {

                                if (i0 >= ps[0] && i0 < ps[0] + src_shape[0] &&
                                    i1 >= ps[1] && i1 < ps[1] + src_shape[1] &&
                                    i2 >= ps[2] && i2 < ps[2] + src_shape[2] &&
                                    i3 >= ps[3] && i3 < ps[3] + src_shape[3]) {

                                    Assert.AreEqual(
                                        src_arr[i0 - ps[0], i1 - ps[1], i2 - ps[2], i3 - ps[3]],
                                        ret_arr[i0, i1, i2, i3]
                                    );
                                }
                                else {
                                    Assert.AreEqual(
                                        dst_arr[i0, i1, i2, i3],
                                        ret_arr[i0, i1, i2, i3]
                                    );
                                }

                            }
                        }
                    }
                }

                return;
            }

            if (src_shape.Ndim == 5) {
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

                return;
            }

            Assert.Fail();
        }

        private static void CheckRegionCopy(Shape src_shape, Shape dst_shape, (int src_offset, int dst_offset, int count)[] region, NdimArray<float> src_arr, NdimArray<float> dst_arr, NdimArray<float> ret_arr) {

            if (src_shape.Ndim == 3) {
                for (int i2 = 0; i2 < dst_shape[2]; i2++) {
                    for (int i1 = 0; i1 < dst_shape[1]; i1++) {
                        for (int i0 = 0; i0 < dst_shape[0]; i0++) {

                            if (i0 >= region[0].dst_offset && i0 < region[0].dst_offset + region[0].count &&
                                i1 >= region[1].dst_offset && i1 < region[1].dst_offset + region[1].count &&
                                i2 >= region[2].dst_offset && i2 < region[2].dst_offset + region[2].count) {

                                Assert.AreEqual(
                                    src_arr[
                                        i0 + region[0].src_offset - region[0].dst_offset,
                                        i1 + region[1].src_offset - region[1].dst_offset,
                                        i2 + region[2].src_offset - region[2].dst_offset],
                                    ret_arr[i0, i1, i2]
                                );
                            }
                            else {
                                Assert.AreEqual(
                                    dst_arr[i0, i1, i2],
                                    ret_arr[i0, i1, i2]
                                );
                            }

                        }
                    }
                }

                return;
            }

            if (src_shape.Ndim == 4) {
                for (int i3 = 0; i3 < dst_shape[3]; i3++) {
                    for (int i2 = 0; i2 < dst_shape[2]; i2++) {
                        for (int i1 = 0; i1 < dst_shape[1]; i1++) {
                            for (int i0 = 0; i0 < dst_shape[0]; i0++) {

                                if (i0 >= region[0].dst_offset && i0 < region[0].dst_offset + region[0].count &&
                                    i1 >= region[1].dst_offset && i1 < region[1].dst_offset + region[1].count &&
                                    i2 >= region[2].dst_offset && i2 < region[2].dst_offset + region[2].count &&
                                    i3 >= region[3].dst_offset && i3 < region[3].dst_offset + region[3].count) {

                                    Assert.AreEqual(
                                        src_arr[
                                            i0 + region[0].src_offset - region[0].dst_offset,
                                            i1 + region[1].src_offset - region[1].dst_offset,
                                            i2 + region[2].src_offset - region[2].dst_offset,
                                            i3 + region[3].src_offset - region[3].dst_offset],
                                        ret_arr[i0, i1, i2, i3]
                                    );
                                }
                                else {
                                    Assert.AreEqual(
                                        dst_arr[i0, i1, i2, i3],
                                        ret_arr[i0, i1, i2, i3]
                                    );
                                }

                            }
                        }
                    }
                }

                return;
            }

            if (src_shape.Ndim == 5) {
                for (int i4 = 0; i4 < dst_shape[4]; i4++) {
                    for (int i3 = 0; i3 < dst_shape[3]; i3++) {
                        for (int i2 = 0; i2 < dst_shape[2]; i2++) {
                            for (int i1 = 0; i1 < dst_shape[1]; i1++) {
                                for (int i0 = 0; i0 < dst_shape[0]; i0++) {

                                    if (i0 >= region[0].dst_offset && i0 < region[0].dst_offset + region[0].count &&
                                        i1 >= region[1].dst_offset && i1 < region[1].dst_offset + region[1].count &&
                                        i2 >= region[2].dst_offset && i2 < region[2].dst_offset + region[2].count &&
                                        i3 >= region[3].dst_offset && i3 < region[3].dst_offset + region[3].count &&
                                        i4 >= region[4].dst_offset && i4 < region[4].dst_offset + region[4].count) {

                                        Assert.AreEqual(
                                            src_arr[
                                                i0 + region[0].src_offset - region[0].dst_offset,
                                                i1 + region[1].src_offset - region[1].dst_offset,
                                                i2 + region[2].src_offset - region[2].dst_offset,
                                                i3 + region[3].src_offset - region[3].dst_offset,
                                                i4 + region[4].src_offset - region[4].dst_offset],
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

                return;
            }

            Assert.Fail();
        }
    }
}
