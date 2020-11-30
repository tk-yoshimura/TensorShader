using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using TensorShader;
using TensorShaderUtil;

namespace TensorShaderUtilTest {
    [TestClass]
    public class NdimArrayUtilTest {
        [TestMethod]
        public void LinspaceTest() {
            NdimArray<double> arr1 = NdimArrayUtil.Linspace(-2d, 2d, 5, endpoint: true);

            Assert.AreEqual(-2d, arr1.Value[0], 1e-8d);
            Assert.AreEqual(-1d, arr1.Value[1], 1e-8d);
            Assert.AreEqual(0d, arr1.Value[2], 1e-8d);
            Assert.AreEqual(+1d, arr1.Value[3], 1e-8d);
            Assert.AreEqual(+2d, arr1.Value[4], 1e-8d);

            NdimArray<double> arr2 = NdimArrayUtil.Linspace(-2d, 2d, 5, endpoint: false);

            Assert.AreEqual(-2.0d, arr2.Value[0], 1e-8d);
            Assert.AreEqual(-1.2d, arr2.Value[1], 1e-8d);
            Assert.AreEqual(-0.4d, arr2.Value[2], 1e-8d);
            Assert.AreEqual(+0.4d, arr2.Value[3], 1e-8d);
            Assert.AreEqual(+1.2d, arr2.Value[4], 1e-8d);

            NdimArray<float> arr3 = NdimArrayUtil.Linspace(-2f, 2f, 5, endpoint: true);

            Assert.AreEqual(-2f, arr3.Value[0], 1e-5f);
            Assert.AreEqual(-1f, arr3.Value[1], 1e-5f);
            Assert.AreEqual(0f, arr3.Value[2], 1e-5f);
            Assert.AreEqual(+1f, arr3.Value[3], 1e-5f);
            Assert.AreEqual(+2f, arr3.Value[4], 1e-5f);

            NdimArray<float> arr4 = NdimArrayUtil.Linspace(-2f, 2f, 5, endpoint: false);

            Assert.AreEqual(-2.0f, arr4.Value[0], 1e-5f);
            Assert.AreEqual(-1.2f, arr4.Value[1], 1e-5f);
            Assert.AreEqual(-0.4f, arr4.Value[2], 1e-5f);
            Assert.AreEqual(+0.4f, arr4.Value[3], 1e-5f);
            Assert.AreEqual(+1.2f, arr4.Value[4], 1e-5f);
        }

        [TestMethod]
        public void SelectTest() {
            NdimArray<double> arr1 = NdimArrayUtil.Linspace(-2d, 2d, 5, endpoint: true);

            NdimArray<double> arr2 = arr1.Select((v) => Math.Sin(v));

            Assert.AreEqual(Math.Sin(-2d), arr2.Value[0]);
            Assert.AreEqual(Math.Sin(-1d), arr2.Value[1]);
            Assert.AreEqual(Math.Sin(0d), arr2.Value[2]);
            Assert.AreEqual(Math.Sin(+1d), arr2.Value[3]);
            Assert.AreEqual(Math.Sin(+2d), arr2.Value[4]);
        }
    }
}
