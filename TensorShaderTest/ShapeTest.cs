using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest {
    [TestClass]
    public class ShapeTest {
        [TestMethod]
        public void ScalarTest() {
            Shape shape = Shape.Scalar();

            Assert.AreEqual(ShapeType.Scalar, shape.Type, "mismatch type");
            Assert.AreEqual(0, shape.Ndim, "mismatch ndim");
            Assert.AreEqual(1, shape.Length, "mismatch length");
            Assert.AreEqual(0, shape.Width, "mismatch width");
            Assert.AreEqual(0, shape.Height, "mismatch height");
            Assert.AreEqual(0, shape.Depth, "mismatch depth");
            Assert.AreEqual(0, shape.Channels, "mismatch channels");
            Assert.AreEqual(0, shape.InChannels, "mismatch inchannels");
            Assert.AreEqual(0, shape.OutChannels, "mismatch outchannels");
            Assert.AreEqual(1, shape.MapSize, "mismatch mapsize");
            Assert.AreEqual(1, shape.DataSize, "mismatch datasize");
            Assert.AreEqual(1, shape.Batch, "mismatch batch");
            Assert.AreEqual("()", shape.ToString(), "mismatch string");

            Assert.IsTrue(shape == Shape.Scalar(), "notequal shape");
        }

        [TestMethod]
        public void VectorTest() {
            Shape shape = Shape.Vector(7);

            Assert.AreEqual(ShapeType.Vector, shape.Type, "mismatch type");
            Assert.AreEqual(1, shape.Ndim, "mismatch ndim");
            Assert.AreEqual(7, shape.Length, "mismatch length");
            Assert.AreEqual(0, shape.Width, "mismatch width");
            Assert.AreEqual(0, shape.Height, "mismatch height");
            Assert.AreEqual(0, shape.Depth, "mismatch depth");
            Assert.AreEqual(7, shape.Channels, "mismatch channels");
            Assert.AreEqual(0, shape.InChannels, "mismatch inchannels");
            Assert.AreEqual(0, shape.OutChannels, "mismatch outchannels");
            Assert.AreEqual(1, shape.MapSize, "mismatch mapsize");
            Assert.AreEqual(7, shape.DataSize, "mismatch datasize");
            Assert.AreEqual(1, shape.Batch, "mismatch batch");
            Assert.AreEqual("(7)", shape.ToString(), "mismatch string");

            Assert.IsTrue(shape == Shape.Vector(7), "notequal shape");
            Assert.IsTrue(shape != Shape.Vector(8), "notequal shape");
        }

        [TestMethod]
        public void Map0DTest() {
            Shape shape = Shape.Map0D(7, 2);

            Assert.AreEqual(ShapeType.Map, shape.Type, "mismatch type");
            Assert.AreEqual(2, shape.Ndim, "mismatch ndim");
            Assert.AreEqual(7 * 2, shape.Length, "mismatch length");
            Assert.AreEqual(0, shape.Width, "mismatch width");
            Assert.AreEqual(0, shape.Height, "mismatch height");
            Assert.AreEqual(0, shape.Depth, "mismatch depth");
            Assert.AreEqual(7, shape.Channels, "mismatch channels");
            Assert.AreEqual(0, shape.InChannels, "mismatch inchannels");
            Assert.AreEqual(0, shape.OutChannels, "mismatch outchannels");
            Assert.AreEqual(1, shape.MapSize, "mismatch mapsize");
            Assert.AreEqual(7, shape.DataSize, "mismatch datasize");
            Assert.AreEqual(2, shape.Batch, "mismatch batch");
            Assert.AreEqual("(7, 2)", shape.ToString(), "mismatch string");

            Assert.IsTrue(shape == Shape.Map0D(7, 2), "notequal shape");
            Assert.IsTrue(shape != Shape.Map0D(7, 1), "notequal shape");
            Assert.IsTrue(shape != Shape.Map0D(8, 1), "notequal shape");
        }

        [TestMethod]
        public void Map1DTest() {
            Shape shape = Shape.Map1D(7, 3, 4);

            Assert.AreEqual(ShapeType.Map, shape.Type, "mismatch type");
            Assert.AreEqual(3, shape.Ndim, "mismatch ndim");
            Assert.AreEqual(7 * 3 * 4, shape.Length, "mismatch length");
            Assert.AreEqual(3, shape.Width, "mismatch width");
            Assert.AreEqual(0, shape.Height, "mismatch height");
            Assert.AreEqual(0, shape.Depth, "mismatch depth");
            Assert.AreEqual(7, shape.Channels, "mismatch channels");
            Assert.AreEqual(0, shape.InChannels, "mismatch inchannels");
            Assert.AreEqual(0, shape.OutChannels, "mismatch outchannels");
            Assert.AreEqual(3, shape.MapSize, "mismatch mapsize");
            Assert.AreEqual(7 * 3, shape.DataSize, "mismatch datasize");
            Assert.AreEqual(4, shape.Batch, "mismatch batch");
            Assert.AreEqual("(7, 3, 4)", shape.ToString(), "mismatch string");

            Assert.IsTrue(shape == Shape.Map1D(7, 3, 4), "notequal shape");
            Assert.IsTrue(shape != Shape.Map1D(7, 3, 1), "notequal shape");
            Assert.IsTrue(shape != Shape.Map1D(8, 3, 1), "notequal shape");
        }

        [TestMethod]
        public void Map2DTest() {
            Shape shape = Shape.Map2D(7, 3, 4, 2);

            Assert.AreEqual(ShapeType.Map, shape.Type, "mismatch type");
            Assert.AreEqual(4, shape.Ndim, "mismatch ndim");
            Assert.AreEqual(7 * 3 * 4 * 2, shape.Length, "mismatch length");
            Assert.AreEqual(3, shape.Width, "mismatch width");
            Assert.AreEqual(4, shape.Height, "mismatch height");
            Assert.AreEqual(0, shape.Depth, "mismatch depth");
            Assert.AreEqual(7, shape.Channels, "mismatch channels");
            Assert.AreEqual(0, shape.InChannels, "mismatch inchannels");
            Assert.AreEqual(0, shape.OutChannels, "mismatch outchannels");
            Assert.AreEqual(3 * 4, shape.MapSize, "mismatch mapsize");
            Assert.AreEqual(7 * 3 * 4, shape.DataSize, "mismatch datasize");
            Assert.AreEqual(2, shape.Batch, "mismatch batch");
            Assert.AreEqual("(7, 3, 4, 2)", shape.ToString(), "mismatch string");

            Assert.IsTrue(shape == Shape.Map2D(7, 3, 4, 2), "notequal shape");
            Assert.IsTrue(shape != Shape.Map2D(7, 3, 4, 1), "notequal shape");
            Assert.IsTrue(shape != Shape.Map2D(8, 3, 4, 1), "notequal shape");
        }

        [TestMethod]
        public void Map3DTest() {
            Shape shape = Shape.Map3D(7, 3, 4, 5, 4);

            Assert.AreEqual(ShapeType.Map, shape.Type, "mismatch type");
            Assert.AreEqual(5, shape.Ndim, "mismatch ndim");
            Assert.AreEqual(7 * 3 * 4 * 5 * 4, shape.Length, "mismatch length");
            Assert.AreEqual(3, shape.Width, "mismatch width");
            Assert.AreEqual(4, shape.Height, "mismatch height");
            Assert.AreEqual(5, shape.Depth, "mismatch depth");
            Assert.AreEqual(7, shape.Channels, "mismatch channels");
            Assert.AreEqual(0, shape.InChannels, "mismatch inchannels");
            Assert.AreEqual(0, shape.OutChannels, "mismatch outchannels");
            Assert.AreEqual(3 * 4 * 5, shape.MapSize, "mismatch mapsize");
            Assert.AreEqual(7 * 3 * 4 * 5, shape.DataSize, "mismatch datasize");
            Assert.AreEqual(4, shape.Batch, "mismatch batch");
            Assert.AreEqual("(7, 3, 4, 5, 4)", shape.ToString(), "mismatch string");

            Assert.IsTrue(shape == Shape.Map3D(7, 3, 4, 5, 4), "notequal shape");
            Assert.IsTrue(shape != Shape.Map3D(7, 3, 4, 5, 1), "notequal shape");
            Assert.IsTrue(shape != Shape.Map3D(8, 3, 4, 5, 1), "notequal shape");
        }

        [TestMethod]
        public void Kernel0DTest() {
            Shape shape = Shape.Kernel0D(5, 7);

            Assert.AreEqual(ShapeType.Kernel, shape.Type, "mismatch type");
            Assert.AreEqual(2, shape.Ndim, "mismatch ndim");
            Assert.AreEqual(5 * 7, shape.Length, "mismatch length");
            Assert.AreEqual(0, shape.Width, "mismatch width");
            Assert.AreEqual(0, shape.Height, "mismatch height");
            Assert.AreEqual(0, shape.Depth, "mismatch depth");
            Assert.AreEqual(5 * 7, shape.Channels, "mismatch channels");
            Assert.AreEqual(5, shape.InChannels, "mismatch inchannels");
            Assert.AreEqual(7, shape.OutChannels, "mismatch outchannels");
            Assert.AreEqual(1, shape.MapSize, "mismatch mapsize");
            Assert.AreEqual(5 * 7, shape.DataSize, "mismatch datasize");
            Assert.AreEqual(1, shape.Batch, "mismatch batch");
            Assert.AreEqual("(5, 7)", shape.ToString(), "mismatch string");

            Assert.IsTrue(shape == Shape.Kernel0D(5, 7), "notequal shape");
            Assert.IsTrue(shape != Shape.Kernel0D(5, 8), "notequal shape");
        }

        [TestMethod]
        public void Kernel1DTest() {
            Shape shape = Shape.Kernel1D(5, 7, 3);

            Assert.AreEqual(ShapeType.Kernel, shape.Type, "mismatch type");
            Assert.AreEqual(3, shape.Ndim, "mismatch ndim");
            Assert.AreEqual(5 * 7 * 3, shape.Length, "mismatch length");
            Assert.AreEqual(3, shape.Width, "mismatch width");
            Assert.AreEqual(0, shape.Height, "mismatch height");
            Assert.AreEqual(0, shape.Depth, "mismatch depth");
            Assert.AreEqual(5 * 7, shape.Channels, "mismatch channels");
            Assert.AreEqual(5, shape.InChannels, "mismatch inchannels");
            Assert.AreEqual(7, shape.OutChannels, "mismatch outchannels");
            Assert.AreEqual(3, shape.MapSize, "mismatch mapsize");
            Assert.AreEqual(5 * 7 * 3, shape.DataSize, "mismatch datasize");
            Assert.AreEqual(1, shape.Batch, "mismatch batch");
            Assert.AreEqual("(5, 7, 3)", shape.ToString(), "mismatch string");

            Assert.IsTrue(shape == Shape.Kernel1D(5, 7, 3), "notequal shape");
            Assert.IsTrue(shape != Shape.Kernel1D(5, 8, 3), "notequal shape");
        }

        [TestMethod]
        public void Kernel2DTest() {
            Shape shape = Shape.Kernel2D(5, 7, 3, 5);

            Assert.AreEqual(ShapeType.Kernel, shape.Type, "mismatch type");
            Assert.AreEqual(4, shape.Ndim, "mismatch ndim");
            Assert.AreEqual(5 * 7 * 3 * 5, shape.Length, "mismatch length");
            Assert.AreEqual(3, shape.Width, "mismatch width");
            Assert.AreEqual(5, shape.Height, "mismatch height");
            Assert.AreEqual(0, shape.Depth, "mismatch depth");
            Assert.AreEqual(5 * 7, shape.Channels, "mismatch channels");
            Assert.AreEqual(5, shape.InChannels, "mismatch inchannels");
            Assert.AreEqual(7, shape.OutChannels, "mismatch outchannels");
            Assert.AreEqual(3 * 5, shape.MapSize, "mismatch mapsize");
            Assert.AreEqual(5 * 7 * 3 * 5, shape.DataSize, "mismatch datasize");
            Assert.AreEqual(1, shape.Batch, "mismatch batch");
            Assert.AreEqual("(5, 7, 3, 5)", shape.ToString(), "mismatch string");

            Assert.IsTrue(shape == Shape.Kernel2D(5, 7, 3, 5), "notequal shape");
            Assert.IsTrue(shape != Shape.Kernel2D(5, 8, 3, 5), "notequal shape");
        }

        [TestMethod]
        public void Kernel3DTest() {
            Shape shape = Shape.Kernel3D(5, 7, 3, 5, 7);

            Assert.AreEqual(ShapeType.Kernel, shape.Type, "mismatch type");
            Assert.AreEqual(5, shape.Ndim, "mismatch ndim");
            Assert.AreEqual(5 * 7 * 3 * 5 * 7, shape.Length, "mismatch length");
            Assert.AreEqual(3, shape.Width, "mismatch width");
            Assert.AreEqual(5, shape.Height, "mismatch height");
            Assert.AreEqual(7, shape.Depth, "mismatch depth");
            Assert.AreEqual(5 * 7, shape.Channels, "mismatch channels");
            Assert.AreEqual(5, shape.InChannels, "mismatch inchannels");
            Assert.AreEqual(7, shape.OutChannels, "mismatch outchannels");
            Assert.AreEqual(3 * 5 * 7, shape.MapSize, "mismatch mapsize");
            Assert.AreEqual(5 * 7 * 3 * 5 * 7, shape.DataSize, "mismatch datasize");
            Assert.AreEqual(1, shape.Batch, "mismatch batch");
            Assert.AreEqual("(5, 7, 3, 5, 7)", shape.ToString(), "mismatch string");

            Assert.IsTrue(shape == Shape.Kernel3D(5, 7, 3, 5, 7), "notequal shape");
            Assert.IsTrue(shape != Shape.Kernel3D(5, 8, 3, 5, 7), "notequal shape");
        }

        [TestMethod]
        public void CompareTest() {
            Shape[] shapelist1 = {
                null,
                Shape.Scalar(), Shape.Vector(5),
                Shape.Map1D(3, 5), Shape.Map2D(3, 3, 5), Shape.Map3D(3, 3, 3, 5),
                Shape.Kernel0D(3, 5), Shape.Kernel1D(3, 3, 5), Shape.Kernel2D(3, 3, 5, 5), Shape.Kernel3D(3, 3, 5, 7, 5),
            };

            Shape[] shapelist2 = {
                null,
                Shape.Scalar(), Shape.Vector(5),
                Shape.Map1D(3, 5), Shape.Map2D(3, 3, 5), Shape.Map3D(3, 3, 3, 5),
                Shape.Kernel0D(3, 5), Shape.Kernel1D(3, 3, 5), Shape.Kernel2D(3, 3, 5, 5), Shape.Kernel3D(3, 3, 5, 7, 5),
            };

            for (int i, j = 0; j < shapelist2.Length; j++) {
                for (i = 0; i < shapelist1.Length; i++) {
                    if (i == j) {
                        Assert.IsTrue(shapelist1[i] == shapelist2[j], $"IsTrue {i}-{j}");
                        Assert.IsFalse(shapelist1[i] != shapelist2[j], $"IsFalse {i}-{j}");
                    }
                    else {
                        Assert.IsTrue(shapelist1[i] != shapelist2[j], $"IsTrue {i}-{j}");
                        Assert.IsFalse(shapelist1[i] == shapelist2[j], $"IsFalse {i}-{j}");
                    }
                }
            }
        }

        [TestMethod]
        public void BadCreateTest() {
            Assert.ThrowsException<ArgumentException>(
                () => { Shape shape = new Shape(ShapeType.Map, new int[] { 2, 4, -1, 3 }); },
                "non positive"
            );

            Assert.ThrowsException<ArgumentOutOfRangeException>(
                () => { Shape shape = new Shape(ShapeType.Map, new int[] { 256, 256, 256, 64 }); },
                "too large"
            );

            Assert.ThrowsException<ArgumentException>(
                () => { Shape shape = new Shape(ShapeType.Scalar, new int[] { 4 }); },
                "scalar"
            );

            Assert.ThrowsException<ArgumentException>(
                () => { Shape shape = new Shape(ShapeType.Vector, new int[] { 4, 4 }); },
                "vector"
            );
        }
    }
}
