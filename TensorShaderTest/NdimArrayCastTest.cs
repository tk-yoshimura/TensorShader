using Microsoft.VisualStudio.TestTools.UnitTesting;

using TensorShader;

namespace TensorShaderTest {
    [TestClass]
    public class NdimArrayCastTest {
        [TestMethod]
        public void ScalarTest() {
            NdimArray<float> arr = 5;

            Assert.AreEqual(Shape.Scalar, arr.Shape);
            CollectionAssert.AreEqual(new float[] { 5 }, arr.Value);
        }

        [TestMethod]
        public void VectorTest() {
            NdimArray<float> arr = new float[]{ 1, 2, 3, 4 };

            Assert.AreEqual(Shape.Vector(4), arr.Shape);
            CollectionAssert.AreEqual(new float[] { 1, 2, 3, 4 }, arr.Value);
        }

        [TestMethod]
        public void Map0Dtype1Test() {
            NdimArray<float> arr = new float[,]{ { 1, 2, 3 } , { 4, 5, 6 } };

            Assert.AreEqual(Shape.Map0D(3, 2), arr.Shape);
            CollectionAssert.AreEqual(new float[] { 1, 2, 3, 4, 5, 6 }, arr.Value);
        }

        [TestMethod]
        public void Map0Dtype2Test() {
            NdimArray<float> arr = new float[][]{ new float[]{ 1, 2, 3 } , new float[]{ 4, 5, 6 } };

            Assert.AreEqual(Shape.Map0D(3, 2), arr.Shape);
            CollectionAssert.AreEqual(new float[] { 1, 2, 3, 4, 5, 6 }, arr.Value);
        }

        [TestMethod]
        public void Map1Dtype1Test() {
            NdimArray<float> arr = new float[,,]{ 
                { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
            };

            Assert.AreEqual(Shape.Map1D(4, 3, 2), arr.Shape);
            CollectionAssert.AreEqual(
                new float[] { 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7 },
                arr.Value
            );
        }

        [TestMethod]
        public void Map1Dtype2Test() {
            NdimArray<float> arr = new float[][,]{ 
                new float[,] { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                new float[,] { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
            };

            Assert.AreEqual(Shape.Map1D(4, 3, 2), arr.Shape);
            CollectionAssert.AreEqual(
                new float[] { 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7 },
                arr.Value
            );
        }

        [TestMethod]
        public void Map2Dtype1Test() {
            NdimArray<float> arr = new float[,,,] {
                { 
                    { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                    { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                },
                { 
                    { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                    { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                }
            };

            Assert.AreEqual(Shape.Map2D(4, 3, 2, 2), arr.Shape);
            CollectionAssert.AreEqual(
                new float[] { 
                    1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7,
                    2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 
                },
                arr.Value
            );
        }

        [TestMethod]
        public void Map2Dtype2Test() {
            NdimArray<float> arr = new float[][,,] {
                new float[,,] { 
                    { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                    { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                },
                new float[,,] { 
                    { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                    { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                }
            };

            Assert.AreEqual(Shape.Map2D(4, 3, 2, 2), arr.Shape);
            CollectionAssert.AreEqual(
                new float[] { 
                    1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7,
                    2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 
                },
                arr.Value
            );
        }

        [TestMethod]
        public void Map3Dtype1Test() {
            NdimArray<float> arr = new float[,,,,] {
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

            Assert.AreEqual(Shape.Map3D(4, 3, 2, 2, 2), arr.Shape);
            CollectionAssert.AreEqual(
                new float[] { 
                    1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7,
                    2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0,  
                    1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0,
                    2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 
                },
                arr.Value
            );
        }

        [TestMethod]
        public void Map3Dtype2Test() {
            NdimArray<float> arr = new float[][,,,] {
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
                }
            };

            Assert.AreEqual(Shape.Map3D(4, 3, 2, 2, 2), arr.Shape);
            CollectionAssert.AreEqual(
                new float[] { 
                    1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7,
                    2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0,  
                    1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0,
                    2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7, 
                },
                arr.Value
            );
        }
    }
}
