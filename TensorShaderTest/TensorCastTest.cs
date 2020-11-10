using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShaderCudaBackend;

namespace TensorShaderTest {
    [TestClass]
    public class TensorCastTest {
        [TestMethod]
        public void ScalarTest() {
            Tensor tensor = 5;

            Assert.AreEqual(Shape.Scalar, tensor.Shape);
            CollectionAssert.AreEqual(new float[] { 5 }, tensor.State);
        }

        [TestMethod]
        public void VectorTest() {
            Tensor tensor = new float[]{ 1, 2, 3, 4 };

            Assert.AreEqual(Shape.Vector(4), tensor.Shape);
            CollectionAssert.AreEqual(new float[] { 1, 2, 3, 4 }, tensor.State);
        }

        [TestMethod]
        public void Map0Dtype1Test() {
            Tensor tensor = new float[,]{ { 1, 2, 3 } , { 4, 5, 6 } };

            Assert.AreEqual(Shape.Map0D(3, 2), tensor.Shape);
            CollectionAssert.AreEqual(new float[] { 1, 2, 3, 4, 5, 6 }, tensor.State);
        }

        [TestMethod]
        public void Map0Dtype2Test() {
            Tensor tensor = ((3, 2), new float[]{ 1, 2, 3, 4, 5, 6 });

            Assert.AreEqual(Shape.Map0D(3, 2), tensor.Shape);
            CollectionAssert.AreEqual(new float[] { 1, 2, 3, 4, 5, 6 }, tensor.State);
        }

        [TestMethod]
        public void Map1Dtype1Test() {
            Tensor tensor = new float[,,]{ 
                { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
            };

            Assert.AreEqual(Shape.Map1D(4, 3, 2), tensor.Shape);
            CollectionAssert.AreEqual(
                new float[] { 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7 },
                tensor.State
            );
        }

        [TestMethod]
        public void Map1Dtype2Test() {
            Tensor tensor = (
                (4, 3, 2),
                new float[] { 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7 }                
            );

            Assert.AreEqual(Shape.Map1D(4, 3, 2), tensor.Shape);
            CollectionAssert.AreEqual(
                new float[] { 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7 },
                tensor.State
            );
        }

        [TestMethod]
        public void Map2Dtype1Test() {
            Tensor tensor = new float[,,,] {
                { 
                    { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                    { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                }
            };

            Assert.AreEqual(Shape.Map2D(4, 3, 2, 1), tensor.Shape);
            CollectionAssert.AreEqual(
                new float[] { 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7 },
                tensor.State
            );
        }

        [TestMethod]
        public void Map2Dtype2Test() {
            Tensor tensor = (
                (4, 3, 2, 1),
                new float[] { 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7 }
            );

            Assert.AreEqual(Shape.Map2D(4, 3, 2, 1), tensor.Shape);
            CollectionAssert.AreEqual(
                new float[] { 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7 },
                tensor.State
            );
        }

        [TestMethod]
        public void Map3Dtype1Test() {
            Tensor tensor = new float[,,,,] {
                {{ 
                    { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                    { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                }}
            };

            Assert.AreEqual(Shape.Map3D(4, 3, 2, 1, 1), tensor.Shape);
            CollectionAssert.AreEqual(
                new float[] { 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7 },
                tensor.State
            );
        }

        [TestMethod]
        public void Map3Dtype2Test() {
            Tensor tensor = (
                (4, 3, 2, 1, 1),
                new float[] { 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7 }
            );

            Assert.AreEqual(Shape.Map3D(4, 3, 2, 1, 1), tensor.Shape);
            CollectionAssert.AreEqual(
                new float[] { 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7 },
                tensor.State
            );
        }
    }
}
