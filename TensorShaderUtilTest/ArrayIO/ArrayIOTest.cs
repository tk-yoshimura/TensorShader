using System;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using TensorShaderUtil.ArrayIO;

namespace TensorShaderUtilTest {
    [TestClass]
    public class ArrayIOTest {
        [TestMethod]
        public void ReadWriteTest() {
            string filepath = "arrayiotest.txt";
            string[] strs;
            int rows, cols;

            ArrayIO.Write1D(filepath, new string[] { "c1", "c2", "c3", "c4" });

            strs = ArrayIO.Read1D(filepath);
            CollectionAssert.AreEqual(new string[] { "c1", "c2", "c3", "c4" }, strs);

            strs = ArrayIO.Read1D(filepath, skip_rows: 1);
            CollectionAssert.AreEqual(new string[] { "c2", "c3", "c4" }, strs);


            ArrayIO.Write2D(filepath, new string[] { "", "c12", "c13", "c24", "c21", "c22", "c23", "c24", "c31", "c32", "c33", "c34" }, 3, 4);

            (strs, rows, cols) = ArrayIO.Read2D(filepath);
            CollectionAssert.AreEqual(new string[] { "", "c12", "c13", "c24", "c21", "c22", "c23", "c24", "c31", "c32", "c33", "c34" }, strs);
            Assert.AreEqual(3, rows);
            Assert.AreEqual(4, cols);

            (strs, rows, cols) = ArrayIO.Read2D(filepath, skip_rows: 1);
            CollectionAssert.AreEqual(new string[] { "c21", "c22", "c23", "c24", "c31", "c32", "c33", "c34" }, strs);
            Assert.AreEqual(2, rows);
            Assert.AreEqual(4, cols);

            (strs, rows, cols) = ArrayIO.Read2D(filepath, skip_cols: 1);
            CollectionAssert.AreEqual(new string[] { "c12", "c13", "c24", "c22", "c23", "c24", "c32", "c33", "c34" }, strs);
            Assert.AreEqual(3, rows);
            Assert.AreEqual(3, cols);

            (strs, rows, cols) = ArrayIO.Read2D(filepath, skip_rows: 1, skip_cols: 1);
            CollectionAssert.AreEqual(new string[] { "c22", "c23", "c24", "c32", "c33", "c34" }, strs);
            Assert.AreEqual(2, rows);
            Assert.AreEqual(3, cols);

            File.Delete(filepath);
        }
    }
}
