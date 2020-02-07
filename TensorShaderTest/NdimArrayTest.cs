using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using TensorShader;

namespace TensorShaderTest {
    [TestClass]
    public class NdimArrayTest {
        [TestMethod]
        public void CreateTest() {
            int[] value = new int[60];

            NdimArray<int> array1 = new NdimArray<int>(value, Shape.Map2D(3, 4, 5, 1), clone_value: false);
            NdimArray<int> array2 = new NdimArray<int>(value, Shape.Map2D(3, 4, 5, 1), clone_value: true);
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
                NdimArray<int> array = new NdimArray<int>(new int[60], Shape.Map2D(3, 4, 5, 2), clone_value: false);
            });

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<int> array = new NdimArray<int>(new int[120], Shape.Map2D(3, 4, 5, 1), clone_value: false);
            });
        }

        [TestMethod]
        public void IndexerTest() {
            int[] value = new int[60];

            NdimArray<int> array = new NdimArray<int>(value, new Shape(ShapeType.Undefined, 3, 4, 5));

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
            NdimArray<int> array = new NdimArray<int>(new int[60], new Shape(ShapeType.Undefined, 3, 4, 5));

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
            NdimArray<float> array = new NdimArray<float>(new float[60], new Shape(ShapeType.Undefined, 3, 4, 5));

            array[0, 0, 0] = 1;

            Tensor tensor = array;

            Assert.AreEqual(new Shape(ShapeType.Undefined, 3, 4, 5), tensor.Shape);
            Assert.AreEqual(60, tensor.Length);
            Assert.AreEqual(1f, tensor.State[0]);

            NdimArray<float> array2 = tensor;

            Assert.AreEqual(new Shape(ShapeType.Undefined, 3, 4, 5), array2.Shape);
            Assert.AreEqual(60, array2.Length);
            Assert.AreEqual(1f, array2[0, 0, 0]);
        }
    }
}
