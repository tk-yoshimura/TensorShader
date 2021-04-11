using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using TensorShaderUtil.DataSplitUtil;

namespace TensorShaderUtilTest.DataSplitUtil {
    [TestClass]
    public class PatchworkCoordTest {

        [TestMethod]
        public void ExecuteTest() {
            PatchworkCoord coord1 = new(32, 32, 0);
            Assert.AreEqual(32, coord1.BlockSize);
            Assert.AreEqual(32, coord1.PatchSize);
            Assert.AreEqual(0, coord1.Margin);
            Assert.AreEqual(false, coord1.NeedsPadding);
            CollectionAssert.AreEqual(new int[] { 0 }, coord1.BlockCoords);
            CollectionAssert.AreEqual(new int[] { 0 }, coord1.PatchCoords);
            CollectionAssert.AreEqual(new int[] { 32 }, coord1.PatchSizes);

            PatchworkCoord coord2 = new(33, 32, 0);
            Assert.AreEqual(32, coord2.BlockSize);
            Assert.AreEqual(32, coord2.PatchSize);
            Assert.AreEqual(0, coord2.Margin);
            Assert.AreEqual(false, coord2.NeedsPadding);
            CollectionAssert.AreEqual(new int[] { 0, 1 }, coord2.BlockCoords);
            CollectionAssert.AreEqual(new int[] { 0, 1 }, coord2.PatchCoords);
            CollectionAssert.AreEqual(new int[] { 32, 32 }, coord2.PatchSizes);

            PatchworkCoord coord3 = new(34, 32, 1);
            Assert.AreEqual(32, coord3.BlockSize);
            Assert.AreEqual(30, coord3.PatchSize);
            Assert.AreEqual(1, coord3.Margin);
            Assert.AreEqual(false, coord3.NeedsPadding);
            CollectionAssert.AreEqual(new int[] { 0, 2 }, coord3.BlockCoords);
            CollectionAssert.AreEqual(new int[] { 0, 3 }, coord3.PatchCoords);
            CollectionAssert.AreEqual(new int[] { 31, 31 }, coord3.PatchSizes);

            PatchworkCoord coord4 = new(64, 32, 0);
            Assert.AreEqual(32, coord4.BlockSize);
            Assert.AreEqual(32, coord4.PatchSize);
            Assert.AreEqual(0, coord4.Margin);
            Assert.AreEqual(false, coord4.NeedsPadding);
            CollectionAssert.AreEqual(new int[] { 0, 32 }, coord4.BlockCoords);
            CollectionAssert.AreEqual(new int[] { 0, 32 }, coord4.PatchCoords);
            CollectionAssert.AreEqual(new int[] { 32, 32 }, coord4.PatchSizes);

            PatchworkCoord coord5 = new(65, 32, 0);
            Assert.AreEqual(32, coord5.BlockSize);
            Assert.AreEqual(32, coord5.PatchSize);
            Assert.AreEqual(0, coord5.Margin);
            Assert.AreEqual(false, coord5.NeedsPadding);
            CollectionAssert.AreEqual(new int[] { 0, 32, 33 }, coord5.BlockCoords);
            CollectionAssert.AreEqual(new int[] { 0, 32, 33 }, coord5.PatchCoords);
            CollectionAssert.AreEqual(new int[] { 32, 32, 32 }, coord5.PatchSizes);

            PatchworkCoord coord6 = new(66, 32, 1);
            Assert.AreEqual(32, coord6.BlockSize);
            Assert.AreEqual(30, coord6.PatchSize);
            Assert.AreEqual(1, coord3.Margin);
            Assert.AreEqual(false, coord6.NeedsPadding);
            CollectionAssert.AreEqual(new int[] { 0, 30, 34 }, coord6.BlockCoords);
            CollectionAssert.AreEqual(new int[] { 0, 31, 35 }, coord6.PatchCoords);
            CollectionAssert.AreEqual(new int[] { 31, 30, 31 }, coord6.PatchSizes);

            PatchworkCoord coord7 = new(66, 32, 2);
            Assert.AreEqual(32, coord7.BlockSize);
            Assert.AreEqual(28, coord7.PatchSize);
            Assert.AreEqual(2, coord7.Margin);
            Assert.AreEqual(false, coord7.NeedsPadding);
            CollectionAssert.AreEqual(new int[] { 0, 28, 34 }, coord7.BlockCoords);
            CollectionAssert.AreEqual(new int[] { 0, 30, 36 }, coord7.PatchCoords);
            CollectionAssert.AreEqual(new int[] { 30, 28, 30 }, coord7.PatchSizes);

            PatchworkCoord coord8 = new(87, 32, 2);
            Assert.AreEqual(32, coord8.BlockSize);
            Assert.AreEqual(28, coord8.PatchSize);
            Assert.AreEqual(2, coord8.Margin);
            Assert.AreEqual(false, coord8.NeedsPadding);
            CollectionAssert.AreEqual(new int[] { 0, 28, 55 }, coord8.BlockCoords);
            CollectionAssert.AreEqual(new int[] { 0, 30, 57 }, coord8.PatchCoords);
            CollectionAssert.AreEqual(new int[] { 30, 28, 30 }, coord8.PatchSizes);

            PatchworkCoord coord9 = new(88, 32, 2);
            Assert.AreEqual(32, coord9.BlockSize);
            Assert.AreEqual(28, coord9.PatchSize);
            Assert.AreEqual(2, coord9.Margin);
            Assert.AreEqual(false, coord9.NeedsPadding);
            CollectionAssert.AreEqual(new int[] { 0, 28, 56 }, coord9.BlockCoords);
            CollectionAssert.AreEqual(new int[] { 0, 30, 58 }, coord9.PatchCoords);
            CollectionAssert.AreEqual(new int[] { 30, 28, 30 }, coord9.PatchSizes);

            PatchworkCoord coord10 = new(89, 32, 2);
            Assert.AreEqual(32, coord10.BlockSize);
            Assert.AreEqual(28, coord10.PatchSize);
            Assert.AreEqual(2, coord10.Margin);
            Assert.AreEqual(false, coord10.NeedsPadding);
            CollectionAssert.AreEqual(new int[] { 0, 28, 56, 57 }, coord10.BlockCoords);
            CollectionAssert.AreEqual(new int[] { 0, 30, 58, 59 }, coord10.PatchCoords);
            CollectionAssert.AreEqual(new int[] { 30, 28, 28, 30 }, coord10.PatchSizes);

            PatchworkCoord coord11 = new(30, 32, 0);
            Assert.AreEqual(32, coord11.BlockSize);
            Assert.AreEqual(32, coord11.PatchSize);
            Assert.AreEqual(0, coord11.Margin);
            Assert.AreEqual(true, coord11.NeedsPadding);
            CollectionAssert.AreEqual(new int[] { 0 }, coord11.BlockCoords);
            CollectionAssert.AreEqual(new int[] { 0 }, coord11.PatchCoords);
            CollectionAssert.AreEqual(new int[] { 32 }, coord11.PatchSizes);
        }

        [TestMethod]
        public void BadCreateTest() {
            Assert.ThrowsException<ArgumentException>(() => {
                PatchworkCoord coord = new(0, 32, 0);
            });

            Assert.ThrowsException<ArgumentException>(() => {
                PatchworkCoord coord = new(32, 0, 0);
            });

            Assert.ThrowsException<ArgumentException>(() => {
                PatchworkCoord coord = new(32, 32, -1);
            });

            Assert.ThrowsException<ArgumentException>(() => {
                PatchworkCoord coord = new(33, 32, 16);
            });
        }
    }
}
