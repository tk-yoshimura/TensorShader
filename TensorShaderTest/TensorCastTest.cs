using Microsoft.VisualStudio.TestTools.UnitTesting;

using TensorShader;

namespace TensorShaderTest {
    [TestClass]
    public class TensorCastTest {
        [TestMethod]
        public void ScalarTest() {
            Tensor tensor = 5;

            Assert.AreEqual(Shape.Scalar, tensor.Shape);
            CollectionAssert.AreEqual(new float[] { 5 }, tensor.State.Value);
        }

        [TestMethod]
        public void VectorTest() {
            Tensor tensor = new float[]{ 1, 2, 3, 4 };

            Assert.AreEqual(Shape.Vector(4), tensor.Shape);
            CollectionAssert.AreEqual(new float[] { 1, 2, 3, 4 }, tensor.State.Value);
        }

        [TestMethod]
        public void Map0Dtype1Test() {
            Tensor tensor = new float[,]{ { 1, 2, 3 } , { 4, 5, 6 } };

            Assert.AreEqual(Shape.Map0D(3, 2), tensor.Shape);
            CollectionAssert.AreEqual(new float[] { 1, 2, 3, 4, 5, 6 }, tensor.State.Value);
        }

        [TestMethod]
        public void Map0Dtype2Test() {
            Tensor tensor = new float[][]{ new float[]{ 1, 2, 3 } , new float[]{ 4, 5, 6 } };

            Assert.AreEqual(Shape.Map0D(3, 2), tensor.Shape);
            CollectionAssert.AreEqual(new float[] { 1, 2, 3, 4, 5, 6 }, tensor.State.Value);
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
                tensor.State.Value
            );
        }

        [TestMethod]
        public void Map1Dtype2Test() {
            Tensor tensor = new float[][,]{ 
                new float[,] { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                new float[,] { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
            };

            Assert.AreEqual(Shape.Map1D(4, 3, 2), tensor.Shape);
            CollectionAssert.AreEqual(
                new float[] { 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7 },
                tensor.State.Value
            );
        }

        [TestMethod]
        public void Map2Dtype1Test() {
            Tensor tensor = new float[,,,] {
                { 
                    { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                    { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                },
                { 
                    { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                    { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                }
            };

            Assert.AreEqual(Shape.Map2D(4, 3, 2, 2), tensor.Shape);
            CollectionAssert.AreEqual(
                new float[] { 
                    1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7,
                    2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 
                },
                tensor.State.Value
            );
        }

        [TestMethod]
        public void Map2Dtype2Test() {
            Tensor tensor = new float[][,,] {
                new float[,,] { 
                    { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                    { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                },
                new float[,,] { 
                    { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                    { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                },
                new float[,,] { 
                    { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                    { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                }
            };

            Assert.AreEqual(Shape.Map2D(4, 3, 2, 3), tensor.Shape);
            CollectionAssert.AreEqual(
                new float[] { 
                    1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7,
                    2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 
                    2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 
                },
                tensor.State.Value
            );
        }

        [TestMethod]
        public void Map3Dtype1Test() {
            Tensor tensor = new float[,,,,] {
                {
                    {
                        { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                        { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                    },
                    {
                        { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                        { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                    }
                },
                {
                    {
                        { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                        { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                    },
                    {
                        { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                        { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                    }
                }
            };

            Assert.AreEqual(Shape.Map3D(4, 3, 2, 2, 2), tensor.Shape);
            CollectionAssert.AreEqual(
                new float[] { 
                    1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7,
                    2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0,  
                    1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0,
                    2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 
                },
                tensor.State.Value
            );
        }

        [TestMethod]
        public void Map3Dtype2Test() {
            Tensor tensor = new float[][,,,] {
                new float[,,,]{
                    {
                        { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                        { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                    },
                    {
                        { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                        { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                    }
                },
                new float[,,,]{
                    {
                        { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                        { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                    },
                    {
                        { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                        { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                    }
                },
                new float[,,,]{
                    {
                        { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                        { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                    },
                    {
                        { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                        { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                    }
                }
            };

            Assert.AreEqual(Shape.Map3D(4, 3, 2, 2, 3), tensor.Shape);
            CollectionAssert.AreEqual(
                new float[] { 
                    1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7,
                    2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0,  
                    1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0,
                    2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 
                    1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0,
                    2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 
                },
                tensor.State.Value
            );
        }
    }
}
