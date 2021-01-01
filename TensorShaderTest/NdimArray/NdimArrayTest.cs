using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest {
    [TestClass]
    public class NdimArrayTest {
        [TestMethod]
        public void CreateTest() {
            int[] value = new int[60];

            NdimArray<int> array1 = new NdimArray<int>(Shape.Map2D(3, 4, 5, 1), value, clone_value: false);
            NdimArray<int> array2 = new NdimArray<int>(Shape.Map2D(3, 4, 5, 1), value, clone_value: true);
            NdimArray<int> array3 = new NdimArray<int>(Shape.Map2D(3, 4, 5, 1));

            value[0] = 1;
            array1[1, 0, 0, 0] = 2;
            array2[1, 0, 0, 0] = 3;

            Assert.AreEqual(60, array1.Length);
            Assert.AreEqual(4, array1.Ndim);
            Assert.AreEqual(ShapeType.Map, array1.Type);
            Assert.AreEqual(1, array1.Value[0]);
            Assert.AreEqual(1, array1[0, 0, 0, 0]);
            Assert.AreEqual(2, array1[1, 0, 0, 0]);
            Assert.AreEqual(2, value[1]);

            Assert.AreEqual(0, array2[0, 0, 0, 0]);
            Assert.AreEqual(3, array2[1, 0, 0, 0]);
            Assert.AreEqual(2, value[1]);

            Assert.IsFalse(array3.Value.Any((i) => i != 0));
        }

        [TestMethod]
        public void BadCreateTest() {
            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<int> array = new NdimArray<int>(Shape.Map2D(3, 4, 5, 2), new int[60], clone_value: false);
            });

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<int> array = new NdimArray<int>(Shape.Map2D(3, 4, 5, 1), new int[120], clone_value: false);
            });
        }

        [TestMethod]
        public void IndexerTest() {
            int[] value = new int[60];

            NdimArray<int> array = new NdimArray<int>(new Shape(ShapeType.Undefined, 3, 4, 5), value, clone_value: false);

            array[0, 0, 0] = 1;
            array[2, 0, 0] = 2;
            array[0, 3, 0] = 3;
            array[0, 0, 4] = 4;
            array[2, 3, 0] = 5;
            array[2, 0, 4] = 6;
            array[0, 3, 4] = 7;
            array[2, 3, 4] = 8;
            array[1, 1, 1] = 9;

            Assert.AreEqual(1, array[0, 0, 0]);
            Assert.AreEqual(2, array[2, 0, 0]);
            Assert.AreEqual(3, array[0, 3, 0]);
            Assert.AreEqual(4, array[0, 0, 4]);
            Assert.AreEqual(5, array[2, 3, 0]);
            Assert.AreEqual(6, array[2, 0, 4]);
            Assert.AreEqual(7, array[0, 3, 4]);
            Assert.AreEqual(8, array[2, 3, 4]);
            Assert.AreEqual(9, array[1, 1, 1]);

            Assert.AreEqual(1, value[0]);
            Assert.AreEqual(2, value[2]);
            Assert.AreEqual(3, value[3 * 3]);
            Assert.AreEqual(4, value[3 * 4 * 4]);
            Assert.AreEqual(5, value[2 + 3 * 3]);
            Assert.AreEqual(6, value[2 + 3 * 4 * 4]);
            Assert.AreEqual(7, value[3 * 3 + 3 * 4 * 4]);
            Assert.AreEqual(8, value[2 + 3 * 3 + 3 * 4 * 4]);
            Assert.AreEqual(9, value[1 + 3 * 1 + 3 * 4 * 1]);
        }

        [TestMethod]
        public void BadIndexerTest() {
            NdimArray<int> array = new NdimArray<int>(new Shape(ShapeType.Undefined, 3, 4, 5), new int[60]);

            Assert.ThrowsException<ArgumentOutOfRangeException>(() => {
                array[3, 3, 4] = 1;
            });

            Assert.ThrowsException<ArgumentOutOfRangeException>(() => {
                int _ = array[3, 3, 4];
            });

            Assert.ThrowsException<ArgumentOutOfRangeException>(() => {
                array[2, 3, 5] = 1;
            });

            Assert.ThrowsException<ArgumentOutOfRangeException>(() => {
                int _ = array[2, 3, 5];
            });

            Assert.ThrowsException<ArgumentException>(() => {
                array[2, 3, 4, 0] = 1;
            });

            Assert.ThrowsException<ArgumentException>(() => {
                int _ = array[2, 3, 4, 0];
            });

            Assert.ThrowsException<ArgumentException>(() => {
                array[2, 3] = 1;
            });

            Assert.ThrowsException<ArgumentException>(() => {
                int _ = array[2, 3];
            });
        }

        [TestMethod]
        public void TensorCastTest() {
            NdimArray<float> array = new Shape(ShapeType.Undefined, 3, 4, 5);

            array[0, 0, 0] = 1;

            Tensor tensor = array;

            Assert.AreEqual(new Shape(ShapeType.Undefined, 3, 4, 5), tensor.Shape);
            Assert.AreEqual(60, tensor.Length);
            Assert.AreEqual(1f, tensor.State.Value[0]);

            NdimArray<float> array2 = tensor;

            Assert.AreEqual(new Shape(ShapeType.Undefined, 3, 4, 5), array2.Shape);
            Assert.AreEqual(60, array2.Length);
            Assert.AreEqual(1f, array2[0, 0, 0]);
        }

        [TestMethod]
        public void JoinTest() {
            NdimArray<int> array1 = new int[,,] { { { 1, 1, 10 }, { 1, 1, 100 }, { 1, 1, 1000 } } };
            NdimArray<int> array2 = new int[,,] { { { 2, 2, 20 }, { 2, 2, 200 }, { 2, 2, 2000 } } };
            NdimArray<int> array3 = new int[,,] { { { 3, 3, 30 }, { 3, 3, 300 }, { 3, 3, 3000 } } };
            NdimArray<int> array4 = new int[,,] { { { 4, 4, 40 }, { 4, 4, 400 }, { 4, 4, 4000 } } };

            NdimArray<int> array_axis0 = NdimArray<int>.Join(
                Axis.Map1D.Channels,
                array1.Reshape(Shape.Map1D(1, 3, 3)), array2.Reshape(Shape.Map1D(1, 3, 3)),
                array3.Reshape(Shape.Map1D(1, 3, 3)), array4.Reshape(Shape.Map1D(1, 3, 3))
            );

            CollectionAssert.AreEqual(
                new int[] {
                    1, 2, 3, 4, 1, 2, 3, 4, 10, 20, 30, 40,
                    1, 2, 3, 4, 1, 2, 3, 4, 100, 200, 300, 400,
                    1, 2, 3, 4, 1, 2, 3, 4, 1000, 2000, 3000, 4000 },
                array_axis0.Value,
                "axis0"
            );

            NdimArray<int> array_axis1 = NdimArray<int>.Join(
                Axis.Map1D.Width,
                array1.Reshape(Shape.Map1D(3, 1, 3)), array2.Reshape(Shape.Map1D(3, 1, 3)),
                array3.Reshape(Shape.Map1D(3, 1, 3)), array4.Reshape(Shape.Map1D(3, 1, 3))
            );

            CollectionAssert.AreEqual(
                new int[] {
                    1, 1, 10, 2, 2, 20, 3, 3, 30, 4, 4, 40,
                    1, 1, 100, 2, 2, 200, 3, 3, 300, 4, 4, 400,
                    1, 1, 1000, 2, 2, 2000, 3, 3, 3000, 4, 4, 4000 },
                array_axis1.Value,
                "axis1"
            );

            NdimArray<int> array_axis2 = NdimArray<int>.Join(Axis.Map1D.Batch, array1, array2, array3, array4);

            CollectionAssert.AreEqual(
                new int[] {
                    1, 1, 10, 1, 1, 100, 1, 1, 1000,
                    2, 2, 20, 2, 2, 200, 2, 2, 2000,
                    3, 3, 30, 3, 3, 300, 3, 3, 3000,
                    4, 4, 40, 4, 4, 400, 4, 4, 4000
                },
                array_axis2.Value,
                "axis2"
            );
        }

        [TestMethod]
        public void SeparateTest() {
            NdimArray<int> array1 = new int[,,] { { { 1, 1, 10 }, { 1, 1, 100 }, { 1, 1, 1000 } } };
            NdimArray<int> array2 = new int[,,] { { { 2, 2, 20 }, { 2, 2, 200 }, { 2, 2, 2000 } } };
            NdimArray<int> array3 = new int[,,] { { { 3, 3, 30 }, { 3, 3, 300 }, { 3, 3, 3000 } } };
            NdimArray<int> array4 = new int[,,] { { { 4, 4, 40 }, { 4, 4, 400 }, { 4, 4, 4000 } } };

            NdimArray<int>[] arrays_axis0 = NdimArray<int>.Join(
                Axis.Map1D.Channels,
                array1.Reshape(Shape.Map1D(1, 3, 3)), array2.Reshape(Shape.Map1D(1, 3, 3)),
                array3.Reshape(Shape.Map1D(1, 3, 3)), array4.Reshape(Shape.Map1D(1, 3, 3))
            ).Separate(Axis.Map1D.Channels);

            CollectionAssert.AreEqual(
                new int[] { 1, 1, 10, 1, 1, 100, 1, 1, 1000 },
                arrays_axis0[0].Value,
                "axis0"
            );

            CollectionAssert.AreEqual(
                new int[] { 2, 2, 20, 2, 2, 200, 2, 2, 2000 },
                arrays_axis0[1].Value,
                "axis0"
            );

            CollectionAssert.AreEqual(
                new int[] { 3, 3, 30, 3, 3, 300, 3, 3, 3000 },
                arrays_axis0[2].Value,
                "axis0"
            );

            CollectionAssert.AreEqual(
                new int[] { 4, 4, 40, 4, 4, 400, 4, 4, 4000 },
                arrays_axis0[3].Value,
                "axis0"
            );

            NdimArray<int>[] arrays_axis1 = NdimArray<int>.Join(
                Axis.Map1D.Width,
                array1.Reshape(Shape.Map1D(3, 1, 3)), array2.Reshape(Shape.Map1D(3, 1, 3)),
                array3.Reshape(Shape.Map1D(3, 1, 3)), array4.Reshape(Shape.Map1D(3, 1, 3))
            ).Separate(Axis.Map1D.Width);

            CollectionAssert.AreEqual(
                new int[] { 1, 1, 10, 1, 1, 100, 1, 1, 1000 },
                arrays_axis1[0].Value,
                "axis1"
            );

            CollectionAssert.AreEqual(
                new int[] { 2, 2, 20, 2, 2, 200, 2, 2, 2000 },
                arrays_axis1[1].Value,
                "axis1"
            );

            CollectionAssert.AreEqual(
                new int[] { 3, 3, 30, 3, 3, 300, 3, 3, 3000 },
                arrays_axis1[2].Value,
                "axis1"
            );

            CollectionAssert.AreEqual(
                new int[] { 4, 4, 40, 4, 4, 400, 4, 4, 4000 },
                arrays_axis1[3].Value,
                "axis1"
            );

            NdimArray<int>[] arrays_axis2 = NdimArray<int>.Join(Axis.Map1D.Batch, array1, array2, array3, array4).Separate(Axis.Map1D.Batch);

            CollectionAssert.AreEqual(
                new int[] { 1, 1, 10, 1, 1, 100, 1, 1, 1000 },
                arrays_axis2[0].Value,
                "axis2"
            );

            CollectionAssert.AreEqual(
                new int[] { 2, 2, 20, 2, 2, 200, 2, 2, 2000 },
                arrays_axis2[1].Value,
                "axis2"
            );

            CollectionAssert.AreEqual(
                new int[] { 3, 3, 30, 3, 3, 300, 3, 3, 3000 },
                arrays_axis2[2].Value,
                "axis2"
            );

            CollectionAssert.AreEqual(
                new int[] { 4, 4, 40, 4, 4, 400, 4, 4, 4000 },
                arrays_axis2[3].Value,
                "axis2"
            );
        }

        [TestMethod]
        public void ConcatTest() {
            NdimArray<int> array1 = new int[,,] { { { 1, 1, 10 }, { 1, 1, 100 }, { 1, 1, 1000 } } };
            NdimArray<int> array2 = new int[,,] { { { 2, 2, 20 }, { 2, 2, 200 }, { 2, 2, 2000 } } };
            NdimArray<int> array3 = new int[,,] { { { 3, 3, 30 }, { 3, 3, 300 }, { 3, 3, 3000 } } };
            NdimArray<int> array4 = new int[,,] { { { 4, 4, 40 }, { 4, 4, 400 }, { 4, 4, 4000 } } };

            NdimArray<int> array = new NdimArray<int>[] { array1, array2, array3, array4 };

            CollectionAssert.AreEqual(
                new int[] {
                    1, 1, 10, 1, 1, 100, 1, 1, 1000,
                    2, 2, 20, 2, 2, 200, 2, 2, 2000,
                    3, 3, 30, 3, 3, 300, 3, 3, 3000,
                    4, 4, 40, 4, 4, 400, 4, 4, 4000
                },
                array.Value
            );
        }

        [TestMethod]
        public void BatchIndexerTest() {
            NdimArray<int> array1 = new int[,,] { { { 1, 1, 10 }, { 1, 1, 100 }, { 1, 1, 1000 } } };
            NdimArray<int> array2 = new int[,,] { { { 2, 2, 20 }, { 2, 2, 200 }, { 2, 2, 2000 } } };
            NdimArray<int> array3 = new int[,,] { { { 3, 3, 30 }, { 3, 3, 300 }, { 3, 3, 3000 } } };
            NdimArray<int> array4 = new int[,,] { { { 4, 4, 40 }, { 4, 4, 400 }, { 4, 4, 4000 } } };
            NdimArray<int> array5 = new int[,,] { { { 5, 5, 50 }, { 5, 5, 500 }, { 5, 5, 5000 } } };

            NdimArray<int> array = new NdimArray<int>[] { array1, array2, array3, array4 };

            array[2] = array5;

            CollectionAssert.AreEqual(
                new int[] {
                    1, 1, 10, 1, 1, 100, 1, 1, 1000,
                    2, 2, 20, 2, 2, 200, 2, 2, 2000,
                    5, 5, 50, 5, 5, 500, 5, 5, 5000,
                    4, 4, 40, 4, 4, 400, 4, 4, 4000
                },
                array.Value
            );

            NdimArray<int> array6 = array[1];

            CollectionAssert.AreEqual(
                new int[] {
                    2, 2, 20, 2, 2, 200, 2, 2, 2000,
                },
                array6.Value
            );
        }

        [TestMethod]
        public void DataListTest() {
            NdimArray<int> array1 = new int[,,] { { { 1, 1, 10 }, { 1, 1, 100 }, { 1, 1, 1000 } } };
            NdimArray<int> array2 = new int[,,] { { { 2, 2, 20 }, { 2, 2, 200 }, { 2, 2, 2000 } } };
            NdimArray<int> array3 = new int[,,] { { { 3, 3, 30 }, { 3, 3, 300 }, { 3, 3, 3000 } } };
            NdimArray<int> array4 = new int[,,] { { { 4, 4, 40 }, { 4, 4, 400 }, { 4, 4, 4000 } } };

            NdimArray<int> array = new NdimArray<int>[] { array1, array2, array3, array4 };
            NdimArray<int>[] arrays = new NdimArray<int>[] { array1, array2, array3, array4 };

            foreach ((int index, NdimArray<int> data) in array.DataList) {
                CollectionAssert.AreEqual(
                    arrays[index].Value,
                    data.Value
                );
            }
        }
    }
}
