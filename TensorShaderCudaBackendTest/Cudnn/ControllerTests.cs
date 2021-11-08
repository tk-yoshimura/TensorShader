﻿using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShaderCudaBackend;
using TensorShaderCudaBackend.Cudnn;

namespace TensorShaderCudaBackendTest {
    [TestClass()]
    public class CudnnControllerTests {
        [TestMethod()]
        public void ControllerTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            CudnnController controller = new(new Stream());
            controller.Dispose();
        }

        [TestMethod()]
        public void ConvolutionForward2DTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            const int n = 4, oc = 3, ic = 2, ih = 8, iw = 6, kh = 3, kw = 5;
            const int oh = ih - kh + 1, ow = iw - kw + 1;

            float[] xs = (new float[n * ic * ih * iw]).Select((_) => 1f).ToArray();
            float[] ys = new float[n * oc * oh * ow + 1];
            float[] ws = (new float[oc * ic * kh * kw]).Select((_) => 1f).ToArray();

            ys[0] = float.NaN;
            ys[1] = float.PositiveInfinity;

            CudaArray<float> xarr = new(xs);
            TensorDescriptor xdesc = new(TensorFormat.NCHW, DataType.Float, n, ic, ih, iw);

            CudaArray<float> yarr = new((ulong)(n * oc * oh * ow + 1));
            TensorDescriptor ydesc = new(TensorFormat.NCHW, DataType.Float, n, oc, oh, ow);

            CudaArray<float> warr = new(ws);
            FilterDescriptor wdesc = new(TensorFormat.NCHW, DataType.Float, oc, ic, kh, kw);

            ConvolutionDescriptor convdesc = new(DataType.Float, (0, 0), (1, 1), (1, 1));

            CudnnController controller = new(new Stream());
            controller.ConvolutionForward(xarr, xdesc, warr, wdesc, convdesc, yarr, ydesc);

            xarr.Read(xs);
            yarr.Read(ys);
            warr.Read(ws);

            for (int i = 0; i < n * ic * ih * iw; i++) {
                Assert.AreEqual(1f, xs[i]);
            }

            for (int i = 0; i < n * oc * oh * ow; i++) {
                Assert.AreEqual(ic * kh * kw, ys[i], 1e-10);
            }
            Assert.AreEqual(0f, ys[n * oc * oh * ow]);

            for (int i = 0; i < oc * ic * kh * kw; i++) {
                Assert.AreEqual(1f, ws[i]);
            }
        }

        [TestMethod()]
        public void ConvolutionBackwardData2DTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            const int n = 4, oc = 3, ic = 2, ih = 8, iw = 6, kh = 3, kw = 5;
            const int oh = ih - kh + 1, ow = iw - kw + 1;

            float[] dxs = new float[n * ic * ih * iw + 1];
            float[] dys = (new float[n * oc * oh * ow]).Select((_) => 1f).ToArray();
            float[] ws = (new float[oc * ic * kh * kw]).Select((_) => 1f).ToArray();

            dxs[0] = float.NaN;
            dxs[1] = float.PositiveInfinity;

            CudaArray<float> dxarr = new((ulong)(n * ic * ih * iw + 1));
            TensorDescriptor dxdesc = new(TensorFormat.NCHW, DataType.Float, n, ic, ih, iw);

            CudaArray<float> dyarr = new(dys);
            TensorDescriptor dydesc = new(TensorFormat.NCHW, DataType.Float, n, oc, oh, ow);

            CudaArray<float> warr = new(ws);
            FilterDescriptor wdesc = new(TensorFormat.NCHW, DataType.Float, oc, ic, kh, kw);

            ConvolutionDescriptor convdesc = new(DataType.Float, (0, 0), (1, 1), (1, 1));

            CudnnController controller = new(new Stream());
            controller.ConvolutionBackwardData(warr, wdesc, dyarr, dydesc, convdesc, dxarr, dxdesc);

            dxarr.Read(dxs);
            dyarr.Read(dys);
            warr.Read(ws);

            for (int i = 0; i < n * ic * ih * iw; i++) {
                Assert.IsTrue(dxs[i] % kh == 0f);
            }
            Assert.AreEqual(0f, dxs[n * ic * ih * iw]);

            for (int i = 0; i < n * oc * oh * ow; i++) {
                Assert.AreEqual(1f, dys[i]);
            }

            for (int i = 0; i < oc * ic * kh * kw; i++) {
                Assert.AreEqual(1f, ws[i]);
            }
        }

        [TestMethod()]
        public void ConvolutionBackwardFilter2DTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            const int n = 4, oc = 3, ic = 2, ih = 8, iw = 6, kh = 3, kw = 5;
            const int oh = ih - kh + 1, ow = iw - kw + 1;

            float[] xs = (new float[n * ic * ih * iw]).Select((_) => 1f).ToArray();
            float[] dys = (new float[n * oc * oh * ow]).Select((_) => 1f).ToArray();
            float[] dws = new float[oc * ic * kh * kw + 1];

            dws[0] = float.NaN;
            dws[1] = float.PositiveInfinity;

            CudaArray<float> xarr = new(xs);
            TensorDescriptor xdesc = new(TensorFormat.NCHW, DataType.Float, n, ic, ih, iw);

            CudaArray<float> dyarr = new(dys);
            TensorDescriptor dydesc = new(TensorFormat.NCHW, DataType.Float, n, oc, oh, ow);

            CudaArray<float> dwarr = new((ulong)(oc * ic * kh * kw + 1));
            FilterDescriptor dwdesc = new(TensorFormat.NCHW, DataType.Float, oc, ic, kh, kw);

            ConvolutionDescriptor convdesc = new(DataType.Float, (0, 0), (1, 1), (1, 1));

            CudnnController controller = new(new Stream());
            controller.ConvolutionBackwardFilter(xarr, xdesc, dyarr, dydesc, convdesc, dwarr, dwdesc);

            xarr.Read(xs);
            dyarr.Read(dys);
            dwarr.Read(dws);

            for (int i = 0; i < n * ic * ih * iw; i++) {
                Assert.AreEqual(1f, xs[i]);
            }

            for (int i = 0; i < n * oc * oh * ow; i++) {
                Assert.AreEqual(1f, dys[i]);
            }

            for (int i = 0; i < oc * ic * kh * kw; i++) {
                Assert.AreEqual(n * oh * ow, dws[i], 1e-10);
            }
            Assert.AreEqual(0f, dws[oc * ic * kh * kw]);
        }
    }
}