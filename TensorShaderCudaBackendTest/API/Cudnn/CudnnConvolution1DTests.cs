using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShaderCudaBackend;
using TensorShaderCudaBackend.Cudnn;

namespace TensorShaderCudaBackendTest.APITest {
    [TestClass()]
    public class CudnnConvolution1DTests {

        [TestMethod()]
        public void ConvolutionNCHWForward1DTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            const int n = 1024, oc = 31, ic = 37, iw = 255, kw = 5;
            const int ow = iw - kw + 1;

            float[] xs = (new float[n * ic * iw]).Select((_) => 1f).ToArray();
            float[] ys = new float[n * oc * ow + 1];
            float[] ws = (new float[oc * ic * kw]).Select((_) => 1f).ToArray();

            ys[0] = float.NaN;
            ys[1] = float.PositiveInfinity;

            CudaArray<float> xarr = new(xs);
            TensorDescriptor xdesc = new(TensorFormat.NCHW, DataType.Float, n, ic, 1, iw);

            CudaArray<float> yarr = new((ulong)(n * oc * ow + 1));
            TensorDescriptor ydesc = new(TensorFormat.NCHW, DataType.Float, n, oc, 1, ow);

            CudaArray<float> warr = new(ws);
            FilterDescriptor wdesc = new(TensorFormat.NCHW, DataType.Float, oc, ic, 1, kw);

            ConvolutionDescriptor convdesc = new(DataType.Float, (0, 0), (1, 1), (1, 1));

            CudnnController controller = new(new Stream());

            ConvolutionFwdAlgoPerf[] prefs = controller.GetConvolutionForwardAlgorithm(xdesc, wdesc, convdesc, ydesc);
            Console.WriteLine($"algos: {prefs.Length}");
            foreach (var pref in prefs) {
                Console.WriteLine(pref.algo);
                Console.WriteLine($"  status : {pref.status}");
                Console.WriteLine($"  asssumed time : {pref.time}");
                Console.WriteLine($"  workspace : {pref.memory}");
                Console.WriteLine($"  math_type : {pref.math_type}");
                Console.WriteLine($"  determinism : {pref.determinism}");
            }

            controller.ConvolutionForward(xarr, xdesc, warr, wdesc, convdesc, yarr, ydesc, prefs[0].algo);

            xarr.Read(xs);
            yarr.Read(ys);
            warr.Read(ws);

            for (int i = 0; i < n * ic * iw; i++) {
                Assert.AreEqual(1f, xs[i]);
            }

            for (int i = 0; i < n * oc * ow; i++) {
                Assert.AreEqual(ic * kw, ys[i], 1e-10);
            }
            Assert.AreEqual(0f, ys[n * oc * ow]);

            for (int i = 0; i < oc * ic * kw; i++) {
                Assert.AreEqual(1f, ws[i]);
            }
        }

        [TestMethod()]
        public void ConvolutionNCHWBackwardData1DTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            const int n = 1024, oc = 31, ic = 37, iw = 255, kw = 5;
            const int ow = iw - kw + 1;

            float[] dxs = new float[n * ic * iw + 1];
            float[] dys = (new float[n * oc * ow]).Select((_) => 1f).ToArray();
            float[] ws = (new float[oc * ic * kw]).Select((_) => 1f).ToArray();

            dxs[0] = float.NaN;
            dxs[1] = float.PositiveInfinity;

            CudaArray<float> dxarr = new((ulong)(n * ic * iw + 1));
            TensorDescriptor dxdesc = new(TensorFormat.NCHW, DataType.Float, n, ic, 1, iw);

            CudaArray<float> dyarr = new(dys);
            TensorDescriptor dydesc = new(TensorFormat.NCHW, DataType.Float, n, oc, 1, ow);

            CudaArray<float> warr = new(ws);
            FilterDescriptor wdesc = new(TensorFormat.NCHW, DataType.Float, oc, ic, 1, kw);

            ConvolutionDescriptor convdesc = new(DataType.Float, (0, 0), (1, 1), (1, 1));

            CudnnController controller = new(new Stream());

            ConvolutionBwdDataAlgoPerf[] prefs = controller.GetConvolutionBackwardDataAlgorithm(wdesc, dydesc, convdesc, dxdesc);
            Console.WriteLine($"algos: {prefs.Length}");
            foreach (var pref in prefs) {
                Console.WriteLine(pref.algo);
                Console.WriteLine($"  status : {pref.status}");
                Console.WriteLine($"  asssumed time : {pref.time}");
                Console.WriteLine($"  workspace : {pref.memory}");
                Console.WriteLine($"  math_type : {pref.math_type}");
                Console.WriteLine($"  determinism : {pref.determinism}");
            }

            controller.ConvolutionBackwardData(warr, wdesc, dyarr, dydesc, convdesc, dxarr, dxdesc, ConvolutionBwdDataAlgo.Algo1);

            dxarr.Read(dxs);
            dyarr.Read(dys);
            warr.Read(ws);

            for (int i = 0; i < n * ic * iw; i++) {
                Assert.IsTrue(dxs[i] % oc == 0f);
            }
            Assert.AreEqual(0f, dxs[n * ic * iw]);

            for (int i = 0; i < n * oc * ow; i++) {
                Assert.AreEqual(1f, dys[i]);
            }

            for (int i = 0; i < oc * ic * kw; i++) {
                Assert.AreEqual(1f, ws[i]);
            }
        }

        [TestMethod()]
        public void ConvolutionNCHWBackwardFilter1DTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            const int n = 1024, oc = 31, ic = 37, iw = 255, kw = 5;
            const int ow = iw - kw + 1;

            float[] xs = (new float[n * ic * iw]).Select((_) => 1f).ToArray();
            float[] dys = (new float[n * oc * ow]).Select((_) => 1f).ToArray();
            float[] dws = new float[oc * ic * kw + 1];

            dws[0] = float.NaN;
            dws[1] = float.PositiveInfinity;

            CudaArray<float> xarr = new(xs);
            TensorDescriptor xdesc = new(TensorFormat.NCHW, DataType.Float, n, ic, 1, iw);

            CudaArray<float> dyarr = new(dys);
            TensorDescriptor dydesc = new(TensorFormat.NCHW, DataType.Float, n, oc, 1, ow);

            CudaArray<float> dwarr = new((ulong)(oc * ic * kw + 1));
            FilterDescriptor dwdesc = new(TensorFormat.NCHW, DataType.Float, oc, ic, 1, kw);

            ConvolutionDescriptor convdesc = new(DataType.Float, (0, 0), (1, 1), (1, 1));

            CudnnController controller = new(new Stream());

            ConvolutionBwdFilterAlgoPerf[] prefs = controller.GetConvolutionBackwardFilterAlgorithm(xdesc, dydesc, convdesc, dwdesc);
            Console.WriteLine($"algos: {prefs.Length}");
            foreach (var pref in prefs) {
                Console.WriteLine(pref.algo);
                Console.WriteLine($"  status : {pref.status}");
                Console.WriteLine($"  asssumed time : {pref.time}");
                Console.WriteLine($"  workspace : {pref.memory}");
                Console.WriteLine($"  math_type : {pref.math_type}");
                Console.WriteLine($"  determinism : {pref.determinism}");
            }

            controller.ConvolutionBackwardFilter(xarr, xdesc, dyarr, dydesc, convdesc, dwarr, dwdesc, prefs[0].algo);

            xarr.Read(xs);
            dyarr.Read(dys);
            dwarr.Read(dws);

            for (int i = 0; i < n * ic * iw; i++) {
                Assert.AreEqual(1f, xs[i]);
            }

            for (int i = 0; i < n * oc * ow; i++) {
                Assert.AreEqual(1f, dys[i]);
            }

            for (int i = 0; i < oc * ic * kw; i++) {
                Assert.AreEqual(n * ow, dws[i], 1e-10);
            }
            Assert.AreEqual(0f, dws[oc * ic * kw]);
        }

        [TestMethod()]
        public void ConvolutionNHWCForward1DTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            const int n = 1024, oc = 31, ic = 37, iw = 255, kw = 5;
            const int ow = iw - kw + 1;

            float[] xs = (new float[n * ic * iw]).Select((_) => 1f).ToArray();
            float[] ys = new float[n * oc * ow + 1];
            float[] ws = (new float[oc * ic * kw]).Select((_) => 1f).ToArray();

            ys[0] = float.NaN;
            ys[1] = float.PositiveInfinity;

            CudaArray<float> xarr = new(xs);
            TensorDescriptor xdesc = new(TensorFormat.NHWC, DataType.Float, n, ic, 1, iw);

            CudaArray<float> yarr = new((ulong)(n * oc * ow + 1));
            TensorDescriptor ydesc = new(TensorFormat.NHWC, DataType.Float, n, oc, 1, ow);

            CudaArray<float> warr = new(ws);
            FilterDescriptor wdesc = new(TensorFormat.NHWC, DataType.Float, oc, ic, 1, kw);

            ConvolutionDescriptor convdesc = new(DataType.Float, (0, 0), (1, 1), (1, 1));

            CudnnController controller = new(new Stream());

            ConvolutionFwdAlgoPerf[] prefs = controller.GetConvolutionForwardAlgorithm(xdesc, wdesc, convdesc, ydesc);
            Console.WriteLine($"algos: {prefs.Length}");
            foreach (var pref in prefs) {
                Console.WriteLine(pref.algo);
                Console.WriteLine($"  status : {pref.status}");
                Console.WriteLine($"  asssumed time : {pref.time}");
                Console.WriteLine($"  workspace : {pref.memory}");
                Console.WriteLine($"  math_type : {pref.math_type}");
                Console.WriteLine($"  determinism : {pref.determinism}");
            }

            controller.ConvolutionForward(xarr, xdesc, warr, wdesc, convdesc, yarr, ydesc, prefs[0].algo);

            xarr.Read(xs);
            yarr.Read(ys);
            warr.Read(ws);

            for (int i = 0; i < n * ic * iw; i++) {
                Assert.AreEqual(1f, xs[i]);
            }

            for (int i = 0; i < n * oc * ow; i++) {
                Assert.AreEqual(ic * kw, ys[i], 1e-10);
            }
            Assert.AreEqual(0f, ys[n * oc * ow]);

            for (int i = 0; i < oc * ic * kw; i++) {
                Assert.AreEqual(1f, ws[i]);
            }
        }

        [TestMethod()]
        public void ConvolutionNHWCBackwardData1DTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            const int n = 1024, oc = 31, ic = 37, iw = 255, kw = 5;
            const int ow = iw - kw + 1;

            float[] dxs = new float[n * ic * iw + 1];
            float[] dys = (new float[n * oc * ow]).Select((_) => 1f).ToArray();
            float[] ws = (new float[oc * ic * kw]).Select((_) => 1f).ToArray();

            dxs[0] = float.NaN;
            dxs[1] = float.PositiveInfinity;

            CudaArray<float> dxarr = new((ulong)(n * ic * iw + 1));
            TensorDescriptor dxdesc = new(TensorFormat.NHWC, DataType.Float, n, ic, 1, iw);

            CudaArray<float> dyarr = new(dys);
            TensorDescriptor dydesc = new(TensorFormat.NHWC, DataType.Float, n, oc, 1, ow);

            CudaArray<float> warr = new(ws);
            FilterDescriptor wdesc = new(TensorFormat.NHWC, DataType.Float, oc, ic, 1, kw);

            ConvolutionDescriptor convdesc = new(DataType.Float, (0, 0), (1, 1), (1, 1));

            CudnnController controller = new(new Stream());

            ConvolutionBwdDataAlgoPerf[] prefs = controller.GetConvolutionBackwardDataAlgorithm(wdesc, dydesc, convdesc, dxdesc);
            Console.WriteLine($"algos: {prefs.Length}");
            foreach (var pref in prefs) {
                Console.WriteLine(pref.algo);
                Console.WriteLine($"  status : {pref.status}");
                Console.WriteLine($"  asssumed time : {pref.time}");
                Console.WriteLine($"  workspace : {pref.memory}");
                Console.WriteLine($"  math_type : {pref.math_type}");
                Console.WriteLine($"  determinism : {pref.determinism}");
            }

            controller.ConvolutionBackwardData(warr, wdesc, dyarr, dydesc, convdesc, dxarr, dxdesc, prefs[0].algo);

            dxarr.Read(dxs);
            dyarr.Read(dys);
            warr.Read(ws);

            for (int i = 0; i < n * ic * iw; i++) {
                Assert.IsTrue(dxs[i] % oc == 0f);
            }
            Assert.AreEqual(0f, dxs[n * ic * iw]);

            for (int i = 0; i < n * oc * ow; i++) {
                Assert.AreEqual(1f, dys[i]);
            }

            for (int i = 0; i < oc * ic * kw; i++) {
                Assert.AreEqual(1f, ws[i]);
            }
        }

        [TestMethod()]
        public void ConvolutionNHWCBackwardFilter1DTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            const int n = 1024, oc = 31, ic = 37, iw = 255, kw = 5;
            const int ow = iw - kw + 1;

            float[] xs = (new float[n * ic * iw]).Select((_) => 1f).ToArray();
            float[] dys = (new float[n * oc * ow]).Select((_) => 1f).ToArray();
            float[] dws = new float[oc * ic * kw + 1];

            dws[0] = float.NaN;
            dws[1] = float.PositiveInfinity;

            CudaArray<float> xarr = new(xs);
            TensorDescriptor xdesc = new(TensorFormat.NHWC, DataType.Float, n, ic, 1, iw);

            CudaArray<float> dyarr = new(dys);
            TensorDescriptor dydesc = new(TensorFormat.NHWC, DataType.Float, n, oc, 1, ow);

            CudaArray<float> dwarr = new((ulong)(oc * ic * kw + 1));
            FilterDescriptor dwdesc = new(TensorFormat.NHWC, DataType.Float, oc, ic, 1, kw);

            ConvolutionDescriptor convdesc = new(DataType.Float, (0, 0), (1, 1), (1, 1));

            CudnnController controller = new(new Stream());

            ConvolutionBwdFilterAlgoPerf[] prefs = controller.GetConvolutionBackwardFilterAlgorithm(xdesc, dydesc, convdesc, dwdesc);
            Console.WriteLine($"algos: {prefs.Length}");
            foreach (var pref in prefs) {
                Console.WriteLine(pref.algo);
                Console.WriteLine($"  status : {pref.status}");
                Console.WriteLine($"  asssumed time : {pref.time}");
                Console.WriteLine($"  workspace : {pref.memory}");
                Console.WriteLine($"  math_type : {pref.math_type}");
                Console.WriteLine($"  determinism : {pref.determinism}");
            }

            controller.ConvolutionBackwardFilter(xarr, xdesc, dyarr, dydesc, convdesc, dwarr, dwdesc, prefs[0].algo);

            xarr.Read(xs);
            dyarr.Read(dys);
            dwarr.Read(dws);

            for (int i = 0; i < n * ic * iw; i++) {
                Assert.AreEqual(1f, xs[i]);
            }

            for (int i = 0; i < n * oc * ow; i++) {
                Assert.AreEqual(1f, dys[i]);
            }

            for (int i = 0; i < oc * ic * kw; i++) {
                Assert.AreEqual(n * ow, dws[i], 1e-10);
            }
            Assert.AreEqual(0f, dws[oc * ic * kw]);
        }
    }
}