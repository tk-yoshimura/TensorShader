using Microsoft.VisualStudio.TestTools.UnitTesting;

using TensorShader;

namespace TensorShaderTest {
    [TestClass]
    public class NdimArrayCastTest {
        [TestMethod]
        public void ScalarTest() {
            NdimArray<float> array = 5;

            Assert.AreEqual(Shape.Scalar, array.Shape);
            CollectionAssert.AreEqual(new float[] { 5 }, array.Value);
            Assert.AreEqual(5, (float)array);
        }

        [TestMethod]
        public void VectorTest() {
            NdimArray<float> array = new float[]{ 1, 2, 3, 4 };

            Assert.AreEqual(Shape.Vector(4), array.Shape);
            CollectionAssert.AreEqual(new float[] { 1, 2, 3, 4 }, array.Value);
            CollectionAssert.AreEqual(new float[] { 1, 2, 3, 4 }, (float[])array);
        }

        [TestMethod]
        public void Map0DTest() {
            NdimArray<float> array = new float[,]{ { 1, 2, 3 } , { 4, 5, 6 } };

            Assert.AreEqual(Shape.Map0D(3, 2), array.Shape);
            CollectionAssert.AreEqual(new float[] { 1, 2, 3, 4, 5, 6 }, array.Value);
        }

        [TestMethod]
        public void Map1DTest() {
            NdimArray<float> array = new float[,,]{ 
                { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
            };

            Assert.AreEqual(Shape.Map1D(4, 3, 2), array.Shape);
            CollectionAssert.AreEqual(
                new float[] { 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7 },
                array.Value
            );
        }

        [TestMethod]
        public void Map2DTest() {
            NdimArray<float> array = new float[,,,] {
                { 
                    { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                    { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                }
            };

            Assert.AreEqual(Shape.Map2D(4, 3, 2, 1), array.Shape);
            CollectionAssert.AreEqual(
                new float[] { 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7 },
                array.Value
            );
        }

        [TestMethod]
        public void Map3DTest() {
            NdimArray<float> array = new float[,,,,] {
                {{ 
                    { { 1, 2, 3, 4 } , { 4, 5, 6, 7 } , { 7, 8, 9, 0 } },
                    { { 2, 3, 4, 1 } , { 5, 6, 7, 4 } , { 8, 9, 0, 7 } },
                }}
            };

            Assert.AreEqual(Shape.Map3D(4, 3, 2, 1, 1), array.Shape);
            CollectionAssert.AreEqual(
                new float[] { 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 0, 2, 3, 4, 1, 5, 6, 7, 4, 8, 9, 0, 7 },
                array.Value
            );
        }
    }
}
