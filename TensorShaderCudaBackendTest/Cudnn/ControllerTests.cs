using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderCudaBackend;
using TensorShaderCudaBackend.Cudnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorShaderCudaBackendTest {
    [TestClass()]
    public class ControllerTests {
        [TestMethod()]
        public void ControllerTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                return;
            }

            Controller controller = new(new Stream());
            controller.Dispose();
        }

        [TestMethod()]
        public void ConvolutionForward2DTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                return;
            }

            int n = 4, oc = 3, ic = 2, ih = 8, iw = 6, kh = 3, kw = 5;
            int oh = ih - kh + 1, ow = iw - kw + 1;

            float[] xs = (new float[n * ic * ih * iw]).Select((_) => 1f).ToArray();
            float[] ys =  new float[n * oc * oh * ow + 1];
            float[] ws = (new float[oc * ic * kh * kw]).Select((_) => 1f).ToArray();

            CudaArray<float> xarr = new(xs);
            TensorDescriptor xdesc = new(TensorFormat.NCHW, DataType.Float, n, ic, ih, iw);

            CudaArray<float> yarr = new((ulong)(n * oc * oh * ow + 1));
            TensorDescriptor ydesc = new(TensorFormat.NCHW, DataType.Float, n, oc, oh, ow);

            CudaArray<float> warr = new(ws);
            FilterDescriptor wdesc = new(TensorFormat.NCHW, DataType.Float, oc, ic, kh, kw);

            ConvolutionDescriptor convdesc = new(DataType.Float, (0, 0), (1, 1), (1, 1));

            Controller controller = new(new Stream());
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
                return;
            }

            int n = 4, oc = 3, ic = 2, ih = 8, iw = 6, kh = 3, kw = 5;
            int oh = ih - kh + 1, ow = iw - kw + 1;

            float[] dxs = new float[n * ic * ih * iw + 1];
            float[] dys = (new float[n * oc * oh * ow]).Select((_) => 1f).ToArray();
            float[] ws = (new float[oc * ic * kh * kw]).Select((_) => 1f).ToArray();

            CudaArray<float> dxarr = new((ulong)(n * ic * ih * iw + 1));
            TensorDescriptor dxdesc = new(TensorFormat.NCHW, DataType.Float, n, ic, ih, iw);

            CudaArray<float> dyarr = new(dys);
            TensorDescriptor dydesc = new(TensorFormat.NCHW, DataType.Float, n, oc, oh, ow);

            CudaArray<float> warr = new(ws);
            FilterDescriptor wdesc = new(TensorFormat.NCHW, DataType.Float, oc, ic, kh, kw);

            ConvolutionDescriptor convdesc = new(DataType.Float, (0, 0), (1, 1), (1, 1));

            Controller controller = new(new Stream());
            controller.ConvolutionBackwardData(warr, wdesc, dyarr, dydesc, convdesc, dxarr, dxdesc);
            
            dxarr.Read(dxs);
            dyarr.Read(dys);
            warr.Read(ws);

            for (int i = 0; i < n * ic * ih * iw; i++) {
                Assert.IsTrue(dxs[i] % 3f == 0f);
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
        public void ConvolutionBackwardFilterTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                return;
            }
            
            Assert.Fail();
        }
    }
}