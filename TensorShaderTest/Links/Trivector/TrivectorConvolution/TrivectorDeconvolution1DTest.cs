using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.TrivectorConvolution {
    [TestClass]
    public class TrivectorDeconvolution1DTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 9, outchannels = 12, kwidth = 3, inwidth = 7;
            int outwidth = inwidth - kwidth + 1, batch = 3;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(inchannels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(outchannels, outwidth, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth), wval);

            VariableField x_actual = xtensor;
            ParameterField w = wtensor;
            ParameterField y = ytensor;

            Field x_expect = TrivectorDeconvolution1D(y, w);
            Field err = x_expect - x_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gy_actual = y.GradTensor.State;
            float[] gw_actual = w.GradTensor.State;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gy_expect, gy_actual, 1e-6f, 1e-5f, $"not equal gy");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int inchannels = 9, outchannels = 12, kwidth = 3, inwidth = 7;
            int outwidth = inwidth - kwidth + 1, batch = 3;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(inchannels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(outchannels, outwidth, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth), wval);

            VariableField x_actual = xtensor;
            ParameterField w = wtensor;
            ParameterField y = ytensor;

            Field yx = TrivectorX(y), yy = TrivectorY(y), yz = TrivectorZ(y);
            Field wr = QuaternionR(w), wi = QuaternionI(w), wj = QuaternionJ(w), wk = QuaternionK(w);
            Field wrr = wr * wr, wii = wi * wi, wjj = wj * wj, wkk = wk * wk;
            Field wri = wr * wi, wrj = wr * wj, wrk = wr * wk, wij = wi * wj, wik = wi * wk, wjk = wj * wk;

            Field xx = Deconvolution1D(yx, (wrr + wii - wjj - wkk)) + 2 * (Deconvolution1D(yy, (wij - wrk)) + Deconvolution1D(yz, (wik + wrj)));
            Field xy = Deconvolution1D(yy, (wrr - wii + wjj - wkk)) + 2 * (Deconvolution1D(yz, (wjk - wri)) + Deconvolution1D(yx, (wij + wrk)));
            Field xz = Deconvolution1D(yz, (wrr - wii - wjj + wkk)) + 2 * (Deconvolution1D(yx, (wik - wrj)) + Deconvolution1D(yy, (wjk + wri)));

            Field x_expect = TrivectorCast(xx, xy, xz);

            Field err = x_expect - x_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gy_actual = y.GradTensor.State;
            float[] gw_actual = w.GradTensor.State;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gy_expect, gy_actual, 1e-6f, 1e-5f, $"not equal gy");
        }

        float[] gy_expect = new float[] {
            -1.842762427e-03f,  -2.107797191e-03f,  -1.243706557e-03f,  -1.363380483e-03f,  -1.574883442e-03f,  -8.920098144e-04f,
            -9.759367924e-04f,  -1.141147410e-03f,  -6.145308934e-04f,  -6.804313558e-04f,  -8.065890954e-04f,  -4.112697935e-04f,
            -3.819821969e-03f,  -4.064785754e-03f,  -3.094001309e-03f,  -2.934857569e-03f,  -3.129904549e-03f,  -2.357944396e-03f,
            -2.198163089e-03f,  -2.350168916e-03f,  -1.750644847e-03f,  -1.609738529e-03f,  -1.725578855e-03f,  -1.272102660e-03f,
            -5.612912093e-03f,  -5.855744426e-03f,  -4.792203319e-03f,  -4.353878063e-03f,  -4.547292659e-03f,  -3.698162104e-03f,
            -3.296434822e-03f,  -3.447245529e-03f,  -2.784860468e-03f,  -2.440582370e-03f,  -2.555603038e-03f,  -2.052298412e-03f,
            -7.692934382e-03f,  -7.941378383e-03f,  -6.776280780e-03f,  -5.911764044e-03f,  -6.108545139e-03f,  -5.173526823e-03f,
            -4.435658308e-03f,  -4.588403291e-03f,  -3.856592988e-03f,  -3.264617177e-03f,  -3.380952840e-03f,  -2.825479274e-03f,
            -1.106299978e-02f,  -1.133623316e-02f,  -1.008501136e-02f,  -8.473278215e-03f,  -8.688452450e-03f,  -7.674380408e-03f,
            -6.319905818e-03f,  -6.485778578e-03f,  -5.684060959e-03f,  -4.602882590e-03f,  -4.728211544e-03f,  -4.114053014e-03f,
            -1.830064917e-02f,  -1.857214455e-02f,  -1.711136832e-02f,  -1.443298464e-02f,  -1.465084868e-02f,  -1.346291107e-02f,
            -1.113052356e-02f,  -1.130193601e-02f,  -1.035668149e-02f,  -8.393265929e-03f,  -8.525406523e-03f,  -7.792679600e-03f,
            -1.905597600e-02f,  -1.930708686e-02f,  -1.771390924e-02f,  -1.499091799e-02f,  -1.519211206e-02f,  -1.389626813e-02f,
            -1.152765996e-02f,  -1.168573324e-02f,  -1.065470359e-02f,  -8.666201896e-03f,  -8.787950427e-03f,  -7.989215615e-03f,
            -2.068964912e-02f,  -2.093817317e-02f,  -1.924054436e-02f,  -1.627688638e-02f,  -1.647608228e-02f,  -1.509359309e-02f,
            -1.251686179e-02f,  -1.267345380e-02f,  -1.157207901e-02f,  -9.409575327e-03f,  -9.530287716e-03f,  -8.676002132e-03f,
            -2.306560133e-02f,  -2.331995354e-02f,  -2.150700115e-02f,  -1.798707006e-02f,  -1.818991861e-02f,  -1.671021306e-02f,
            -1.371236114e-02f,  -1.387125320e-02f,  -1.269274751e-02f,  -1.024147458e-02f,  -1.036395733e-02f,  -9.454604497e-03f,
            -2.789073787e-02f,  -2.816995411e-02f,  -2.625583529e-02f,  -2.161704611e-02f,  -2.183842217e-02f,  -2.026239098e-02f,
            -1.634342411e-02f,  -1.651565795e-02f,  -1.524963447e-02f,  -1.206987188e-02f,  -1.220166145e-02f,  -1.121756577e-02f,
            -3.475853591e-02f,  -3.503649190e-02f,  -3.297903009e-02f,  -2.750258880e-02f,  -2.772681392e-02f,  -2.603381232e-02f,
            -2.128511033e-02f,  -2.146272460e-02f,  -2.009883209e-02f,  -1.610610050e-02f,  -1.624422395e-02f,  -1.517408941e-02f,
            -3.429213002e-02f,  -3.454938797e-02f,  -3.233381716e-02f,  -2.704697841e-02f,  -2.725431956e-02f,  -2.543459186e-02f,
            -2.085715683e-02f,  -2.102129757e-02f,  -1.955876233e-02f,  -1.572266526e-02f,  -1.585032200e-02f,  -1.470632857e-02f,
            -3.576638614e-02f,  -3.602060191e-02f,  -3.368888541e-02f,  -2.819989471e-02f,  -2.840487190e-02f,  -2.648902407e-02f,
            -2.173728875e-02f,  -2.189966206e-02f,  -2.035929755e-02f,  -1.637856828e-02f,  -1.650497239e-02f,  -1.529970585e-02f,
            -3.843826828e-02f,  -3.869852869e-02f,  -3.623772153e-02f,  -3.006237607e-02f,  -3.027129208e-02f,  -2.824689929e-02f,
            -2.298906397e-02f,  -2.315410312e-02f,  -2.152890202e-02f,  -1.721833198e-02f,  -1.734696181e-02f,  -1.608372972e-02f,
            -4.471847597e-02f,  -4.500367506e-02f,  -4.242665922e-02f,  -3.476081400e-02f,  -3.498839189e-02f,  -3.285040155e-02f,
            -2.636694240e-02f,  -2.654553732e-02f,  -2.481520798e-02f,  -1.953686118e-02f,  -1.967511136e-02f,  -1.832107852e-02f,
        };

        float[] gw_expect = new float[] {
            -8.150946873e-02f,  -7.683236991e-02f,  -8.032103527e-02f,  -8.387079219e-02f,  -8.561092045e-02f,  -8.086958005e-02f,
            -8.435634020e-02f,  -8.782832501e-02f,  -8.919300970e-02f,  -8.439128947e-02f,  -8.787316940e-02f,  -9.126850746e-02f,
            -7.655094486e-02f,  -7.202475620e-02f,  -7.532060071e-02f,  -7.872218169e-02f,  -8.022091272e-02f,  -7.563097087e-02f,
            -7.891941657e-02f,  -8.224432273e-02f,  -8.337044661e-02f,  -7.872052546e-02f,  -8.199879988e-02f,  -8.524818394e-02f,
            -7.124761638e-02f,  -6.688312046e-02f,  -6.997816731e-02f,  -7.321517822e-02f,  -7.445267551e-02f,  -7.002453027e-02f,
            -7.310717939e-02f,  -7.626877700e-02f,  -7.713690202e-02f,  -7.264880410e-02f,  -7.571646965e-02f,  -7.880383404e-02f,
            -6.559948328e-02f,  -6.140746269e-02f,  -6.429373507e-02f,  -6.734978178e-02f,  -6.830620884e-02f,  -6.405025826e-02f,
            -6.691962867e-02f,  -6.990168782e-02f,  -7.049237595e-02f,  -6.617612540e-02f,  -6.902617869e-02f,  -7.193545774e-02f,
            -6.330276222e-02f,  -5.929190499e-02f,  -6.196498336e-02f,  -6.458807482e-02f,  -6.475815505e-02f,  -6.069008068e-02f,
            -6.334699791e-02f,  -6.590338875e-02f,  -6.574488609e-02f,  -6.162396317e-02f,  -6.426122265e-02f,  -6.675213690e-02f,
            -5.671763685e-02f,  -5.287829609e-02f,  -5.533371666e-02f,  -5.778065727e-02f,  -5.767323945e-02f,  -5.377645727e-02f,
            -5.621095992e-02f,  -5.859283235e-02f,  -5.816113916e-02f,  -5.421120225e-02f,  -5.662149436e-02f,  -5.893955408e-02f,
            -4.972427604e-02f,  -4.606713300e-02f,  -4.829713405e-02f,  -5.055140350e-02f,  -5.014617804e-02f,  -4.643096493e-02f,
            -4.863580348e-02f,  -5.082682588e-02f,  -5.010210809e-02f,  -4.633304124e-02f,  -4.850961137e-02f,  -5.063867365e-02f,
            -4.232267978e-02f,  -3.885841571e-02f,  -4.085523553e-02f,  -4.290031351e-02f,  -4.217697081e-02f,  -3.865360368e-02f,
            -4.062152858e-02f,  -4.260536935e-02f,  -4.156779289e-02f,  -3.798948013e-02f,  -3.992557368e-02f,  -4.184949560e-02f,
            -3.751979238e-02f,  -3.415638480e-02f,  -3.593198061e-02f,  -3.763477278e-02f,  -3.609178632e-02f,  -3.268477296e-02f,
            -3.442861644e-02f,  -3.608026342e-02f,  -3.429161522e-02f,  -3.084545213e-02f,  -3.255365997e-02f,  -3.415528655e-02f,
            -2.866365402e-02f,  -2.548064186e-02f,  -2.702288388e-02f,  -2.853194757e-02f,  -2.670161464e-02f,  -2.347453497e-02f,
            -2.498081645e-02f,  -2.644045472e-02f,  -2.436934328e-02f,  -2.110258883e-02f,  -2.256920047e-02f,  -2.398056882e-02f,
            -1.931070033e-02f,  -1.631773016e-02f,  -1.762018136e-02f,  -1.891945089e-02f,  -1.678259025e-02f,  -1.374472888e-02f,
            -1.500746220e-02f,  -1.625921023e-02f,  -1.388693372e-02f,  -1.080852119e-02f,  -1.202798544e-02f,  -1.323339814e-02f,
            -9.460931295e-03f,  -6.667649713e-03f,  -7.723873050e-03f,  -8.797282755e-03f,  -6.334713148e-03f,  -3.495354703e-03f,
            -4.508553713e-03f,  -5.536529969e-03f,  -2.844386564e-03f,  3.675078970e-05f,  -9.300148580e-04f,  -1.913774527e-03f,
        };
    }
}
