using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest {
    [TestClass]
    public class NdimArrayTrimmingTest {
        [TestMethod]
        public void TrimmingTest() {
            Random random = new Random();

            Shape shape = new Shape(ShapeType.Map, new int[] { 5, 9, 7, 8, 9 });
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
                float[] v = (new float[shape.Length]).Select((v) => (float)random.NextDouble()).ToArray();

                NdimArray<float> arr = new NdimArray<float>(shape, v);
                NdimArray<float> new_arr = NdimArray<float>.Trimming(arr, trims);

                Shape new_shape = new_arr.Shape;

                for (int i4 = 0; i4 < new_shape[4]; i4++) {
                    for (int i3 = 0; i3 < new_shape[3]; i3++) {
                        for (int i2 = 0; i2 < new_shape[2]; i2++) {
                            for (int i1 = 0; i1 < new_shape[1]; i1++) {
                                for (int i0 = 0; i0 < new_shape[0]; i0++) {

                                    Assert.AreEqual(
                                        arr[i4 + trims[4].ta, i3 + trims[3].ta, i2 + trims[2].ta, i1 + trims[1].ta, i0 + trims[0].ta], 
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
