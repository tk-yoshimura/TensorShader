using System;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderUtil;

namespace TensorShaderUtilTest {
    [TestClass]
    public class Crc32Test {
        [TestMethod]
        public void ExecuteTest() {
            string str = "abcdefghijklmnopqrstuvwxyz\0";
            byte[] data = new UTF8Encoding().GetBytes(str);

            UInt32 crc = Crc32.ComputeHash(data, 0, data.Length);

            Assert.AreEqual(1738409964ul, crc);
        }
    }
}
