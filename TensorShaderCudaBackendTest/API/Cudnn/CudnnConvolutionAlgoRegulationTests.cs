using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShaderCudaBackend;
using TensorShaderCudaBackend.Cudnn;

namespace TensorShaderCudaBackendTest.APITest {
    [TestClass()]
    public class CudnnConvolutionAlgoRegulationTests {

        [TestMethod()]
        public void ForwardAlgoRegulationTest() {
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

            CudnnController controller = new(new Stream(), workspace_limit_bytes: 16);

            ConvolutionFwdAlgoPerf[] prefs = controller.EnumConvolutionForwardAlgorithm(xdesc, wdesc, convdesc, ydesc);
            Console.WriteLine($"algos: {prefs.Length}");
            foreach (var pref in prefs) {
                Console.WriteLine(pref.algo);
                Console.WriteLine($"  status : {pref.status}");
                Console.WriteLine($"  asssumed time : {pref.time}");
                Console.WriteLine($"  workspace : {pref.memory}");
                Console.WriteLine($"  math_type : {pref.math_type}");
                Console.WriteLine($"  determinism : {pref.determinism}");
            }

            ConvolutionFwdAlgo algo = prefs.OrderByDescending((pref) => pref.memory).First().algo;

            controller.ConvolutionForward(xarr, xdesc, warr, wdesc, convdesc, yarr, ydesc, algo);
        }


        [TestMethod()]
        public void BackwardDataAlgoRegulationTest() {
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

            CudnnController controller = new(new Stream(), workspace_limit_bytes: 16);

            ConvolutionBwdDataAlgoPerf[] prefs = controller.EnumConvolutionBackwardDataAlgorithm(wdesc, dydesc, convdesc, dxdesc);
            Console.WriteLine($"algos: {prefs.Length}");
            foreach (var pref in prefs) {
                Console.WriteLine(pref.algo);
                Console.WriteLine($"  status : {pref.status}");
                Console.WriteLine($"  asssumed time : {pref.time}");
                Console.WriteLine($"  workspace : {pref.memory}");
                Console.WriteLine($"  math_type : {pref.math_type}");
                Console.WriteLine($"  determinism : {pref.determinism}");
            }

            ConvolutionBwdDataAlgo algo = prefs.OrderByDescending((pref) => pref.memory).First().algo;

            controller.ConvolutionBackwardData(warr, wdesc, dyarr, dydesc, convdesc, dxarr, dxdesc, algo);
        }

        [TestMethod()]
        public void BackwardFilterAlgoRegulationTest() {
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

            CudnnController controller = new(new Stream(), workspace_limit_bytes: 16);

            ConvolutionBwdFilterAlgoPerf[] prefs = controller.EnumConvolutionBackwardFilterAlgorithm(xdesc, dydesc, convdesc, dwdesc);
            Console.WriteLine($"algos: {prefs.Length}");
            foreach (var pref in prefs) {
                Console.WriteLine(pref.algo);
                Console.WriteLine($"  status : {pref.status}");
                Console.WriteLine($"  asssumed time : {pref.time}");
                Console.WriteLine($"  workspace : {pref.memory}");
                Console.WriteLine($"  math_type : {pref.math_type}");
                Console.WriteLine($"  determinism : {pref.determinism}");
            }

            ConvolutionBwdFilterAlgo algo = prefs.OrderByDescending((pref) => pref.memory).First().algo;

            controller.ConvolutionBackwardFilter(xarr, xdesc, dyarr, dydesc, convdesc, dwarr, dwdesc, algo);
        }
    }
}