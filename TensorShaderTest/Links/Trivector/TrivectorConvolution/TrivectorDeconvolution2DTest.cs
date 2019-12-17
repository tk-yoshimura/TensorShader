using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.TrivectorConvolution {
    [TestClass]
    public class TrivectorDeconvolution2DTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 9, outchannels = 12, kwidth = 3, kheight = 5, stride = 2, inwidth = 7, inheight = 8;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, batch = 3;

            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * outchannels * inchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight), wval);

            VariableField x_actual = xtensor;
            ParameterField w = wtensor;
            ParameterField y = ytensor;

            Field x_expect = TrivectorDeconvolution2D(y, w, stride, Shape.Map2D(inchannels, inwidth, inheight, batch));
            Field err = x_expect - x_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gy_actual = y.GradTensor.State;
            float[] gw_actual = w.GradTensor.State;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gy_expect, gy_actual, 1e-6f, 1e-5f, $"not equal gy");
        }

        float[] gy_expect = new float[] {
            1.895854304e+00f,  2.135468180e+00f,  2.482405765e+00f,  1.792402931e+00f,  2.020535440e+00f,  2.350245408e+00f,
            1.691442903e+00f,  1.908431721e+00f,  2.221459049e+00f,  1.592974219e+00f,  1.799157023e+00f,  2.096046688e+00f,
            5.448086146e+00f,  5.755632756e+00f,  6.169444703e+00f,  5.167389709e+00f,  5.460493786e+00f,  5.853674744e+00f,
            4.894530316e+00f,  5.173605198e+00f,  5.546812577e+00f,  4.629507965e+00f,  4.894966993e+00f,  5.248858203e+00f,
            5.895914744e+00f,  6.138928316e+00f,  6.495884458e+00f,  5.622196489e+00f,  5.854097255e+00f,  6.193330417e+00f,
            5.354584878e+00f,  5.575675362e+00f,  5.897731041e+00f,  5.093079912e+00f,  5.303662636e+00f,  5.609086330e+00f,
            3.421574548e+00f,  3.662657765e+00f,  4.049003049e+00f,  3.364433383e+00f,  3.595955280e+00f,  3.965112989e+00f,
            3.302658128e+00f,  3.524813865e+00f,  3.877190092e+00f,  3.236248783e+00f,  3.449233521e+00f,  3.785234358e+00f,
            8.252226065e+00f,  8.556634009e+00f,  9.015371016e+00f,  8.011004985e+00f,  8.303512204e+00f,  8.741842308e+00f,
            7.768721975e+00f,  8.049569397e+00f,  8.467974970e+00f,  7.525377035e+00f,  7.794805589e+00f,  8.193769002e+00f,
            7.072394533e+00f,  7.308846111e+00f,  7.708754931e+00f,  6.902008101e+00f,  7.129534080e+00f,  7.511813184e+00f,
            6.728175105e+00f,  6.946944668e+00f,  7.311998376e+00f,  6.550895546e+00f,  6.761077874e+00f,  7.109310505e+00f,
            4.731665185e+00f,  4.971052479e+00f,  5.401242249e+00f,  4.497337664e+00f,  4.725247414e+00f,  5.133426574e+00f,
            4.267193902e+00f,  4.483964164e+00f,  4.870840128e+00f,  4.041233897e+00f,  4.247202731e+00f,  4.613482914e+00f,
            1.317367707e+01f,  1.348100362e+01f,  1.397908079e+01f,  1.254963160e+01f,  1.284251944e+01f,  1.331458564e+01f,
            1.194051036e+01f,  1.221937331e+01f,  1.266628327e+01f,  1.134631336e+01f,  1.161156523e+01f,  1.203417368e+01f,
            8.965685797e+00f,  9.208461321e+00f,  9.657839353e+00f,  8.588139897e+00f,  8.819804908e+00f,  9.246288263e+00f,
            8.215867438e+00f,  8.436724842e+00f,  8.841012282e+00f,  7.848868419e+00f,  8.059221122e+00f,  8.442011409e+00f,
            6.320566446e+00f,  6.561419641e+00f,  7.067891529e+00f,  6.272732530e+00f,  6.504021347e+00f,  6.987206922e+00f,
            6.211747452e+00f,  6.433667434e+00f,  6.894093949e+00f,  6.137611212e+00f,  6.350357902e+00f,  6.788552608e+00f,
            1.569782314e+01f,  1.600202392e+01f,  1.659042432e+01f,  1.529856792e+01f,  1.559086320e+01f,  1.615198684e+01f,
            1.489298889e+01f,  1.517362007e+01f,  1.570809503e+01f,  1.448108605e+01f,  1.475029454e+01f,  1.525874891e+01f,
            9.626393257e+00f,  9.862630866e+00f,  1.039145073e+01f,  9.511405606e+00f,  9.738713030e+00f,  1.024366341e+01f,
            9.382589155e+00f,  9.601135958e+00f,  1.008273315e+01f,  9.239943903e+00f,  9.449899649e+00f,  9.908659970e+00f,
            7.567476066e+00f,  7.806636778e+00f,  8.320078734e+00f,  7.202272398e+00f,  7.429959387e+00f,  7.916607739e+00f,
            6.842944901e+00f,  7.059496608e+00f,  7.520221208e+00f,  6.489493575e+00f,  6.695248440e+00f,  7.130919140e+00f,
            2.089926800e+01f,  2.120637448e+01f,  2.178871688e+01f,  1.993187349e+01f,  2.022454509e+01f,  2.077549654e+01f,
            1.898649041e+01f,  1.926514142e+01f,  1.978575397e+01f,  1.806311876e+01f,  1.832816347e+01f,  1.881948915e+01f,
            1.203545685e+01f,  1.227799433e+01f,  1.281979425e+01f,  1.155408330e+01f,  1.178551256e+01f,  1.229924611e+01f,
            1.107715000e+01f,  1.129777432e+01f,  1.178429352e+01f,  1.060465693e+01f,  1.081477961e+01f,  1.127493649e+01f,
            9.219558343e+00f,  9.460181517e+00f,  1.008678001e+01f,  9.181031677e+00f,  9.412087415e+00f,  1.000930086e+01f,
            9.120836776e+00f,  9.342521004e+00f,  9.910997806e+00f,  9.038973641e+00f,  9.251482283e+00f,  9.791870858e+00f,
            2.314342021e+01f,  2.344741383e+01f,  2.416547763e+01f,  2.258613085e+01f,  2.287821420e+01f,  2.356213136e+01f,
            2.201725581e+01f,  2.229767075e+01f,  2.294821509e+01f,  2.143679506e+01f,  2.170578350e+01f,  2.232372882e+01f,
            1.218039198e+01f,  1.241641562e+01f,  1.307414653e+01f,  1.212080311e+01f,  1.234789198e+01f,  1.297551363e+01f,
            1.203700320e+01f,  1.225532725e+01f,  1.285346793e+01f,  1.192899226e+01f,  1.213872142e+01f,  1.270800943e+01f,
        };

        float[] gw_expect = new float[] {
            8.997130580e+00f,  9.083780282e+00f,  8.943997543e+00f,  8.889080697e+00f,  8.717631891e+00f,  8.804353690e+00f,
            8.665215511e+00f,  8.610716953e+00f,  8.442145634e+00f,  8.528937267e+00f,  8.390441958e+00f,  8.336359659e+00f,
            9.033472746e+00f,  9.122593991e+00f,  8.978355434e+00f,  8.923327887e+00f,  8.751928295e+00f,  8.841053578e+00f,
            8.697581080e+00f,  8.642974518e+00f,  8.474455706e+00f,  8.563583706e+00f,  8.420874061e+00f,  8.366686416e+00f,
            9.063484121e+00f,  9.154895010e+00f,  9.006448299e+00f,  8.951360420e+00f,  8.780059190e+00f,  8.871408174e+00f,
            8.723845967e+00f,  8.669181529e+00f,  8.500764565e+00f,  8.592051249e+00f,  8.445368848e+00f,  8.391125741e+00f,
            9.087164705e+00f,  9.180683340e+00f,  9.028276138e+00f,  8.973178294e+00f,  8.802024575e+00f,  8.895417477e+00f,
            8.744010174e+00f,  8.689337985e+00f,  8.521072209e+00f,  8.614339896e+00f,  8.463926320e+00f,  8.409677636e+00f,
            2.167235341e+00f,  2.248975523e+00f,  2.136938675e+00f,  2.096893477e+00f,  2.018627635e+00f,  2.100481284e+00f,
            1.988753852e+00f,  1.948968578e+00f,  1.872459063e+00f,  1.954422467e+00f,  1.843006958e+00f,  1.803480387e+00f,
            2.177353753e+00f,  2.260189517e+00f,  2.146360188e+00f,  2.106151591e+00f,  2.027919642e+00f,  2.110829500e+00f,
            1.997381286e+00f,  1.957434671e+00f,  1.880962704e+00f,  1.963943488e+00f,  1.850878012e+00f,  1.811192134e+00f,
            2.185692312e+00f,  2.269515582e+00f,  2.154037619e+00f,  2.113702160e+00f,  2.035540655e+00f,  2.119399896e+00f,
            2.004372941e+00f,  1.964301376e+00f,  1.887903600e+00f,  1.971796186e+00f,  1.857220989e+00f,  1.817412045e+00f,
            2.192251019e+00f,  2.276953718e+00f,  2.159970967e+00f,  2.119545184e+00f,  2.041490672e+00f,  2.126192474e+00f,
            2.009728818e+00f,  1.969568691e+00f,  1.893281750e+00f,  1.977980561e+00f,  1.862035890e+00f,  1.822140121e+00f,
            6.811834000e+00f,  6.875129875e+00f,  6.770650031e+00f,  6.724581578e+00f,  6.575406931e+00f,  6.639001181e+00f,
            6.534821223e+00f,  6.489124041e+00f,  6.342723359e+00f,  6.406610402e+00f,  6.302733608e+00f,  6.257405660e+00f,
            6.833285987e+00f,  6.898375891e+00f,  6.790424340e+00f,  6.744393576e+00f,  6.595370218e+00f,  6.660698821e+00f,
            6.553156641e+00f,  6.507499162e+00f,  6.361255158e+00f,  6.426817763e+00f,  6.319686798e+00f,  6.274400525e+00f,
            6.848741870e+00f,  6.915452303e+00f,  6.804263877e+00f,  6.758321636e+00f,  6.609504442e+00f,  6.676395962e+00f,
            6.565723426e+00f,  6.520156231e+00f,  6.374123904e+00f,  6.441192685e+00f,  6.331036465e+00f,  6.285842194e+00f,
            6.858201647e+00f,  6.926359110e+00f,  6.812168643e+00f,  6.766365759e+00f,  6.617809606e+00f,  6.686092606e+00f,
            6.572521578e+00f,  6.527095247e+00f,  6.381329598e+00f,  6.449735168e+00f,  6.336782610e+00f,  6.291730668e+00f,
            1.576686158e+00f,  1.659090355e+00f,  1.547817705e+00f,  1.507171674e+00f,  1.424265881e+00f,  1.506669498e+00f,
            1.395967362e+00f,  1.355666567e+00f,  1.274979932e+00f,  1.357379728e+00f,  1.247247884e+00f,  1.207290295e+00f,
            1.575502353e+00f,  1.658506753e+00f,  1.546167787e+00f,  1.505497362e+00f,  1.422806786e+00f,  1.505772203e+00f,
            1.394077038e+00f,  1.353753096e+00f,  1.273287222e+00f,  1.356211115e+00f,  1.245158375e+00f,  1.205178846e+00f,
            1.572966560e+00f,  1.656457038e+00f,  1.543202109e+00f,  1.502548959e+00f,  1.420125474e+00f,  1.503540203e+00f,
            1.391000028e+00f,  1.350694366e+00f,  1.270501082e+00f,  1.353838208e+00f,  1.242010275e+00f,  1.202049992e+00f,
            1.569078778e+00f,  1.652941210e+00f,  1.538920671e+00f,  1.498326465e+00f,  1.416221947e+00f,  1.499973499e+00f,
            1.386736332e+00f,  1.346490379e+00f,  1.266621512e+00f,  1.350261007e+00f,  1.237803583e+00f,  1.197903734e+00f,
            -1.487076523e+00f,  -1.408572291e+00f,  -1.502170038e+00f,  -1.533137600e+00f,  -1.559133146e+00f,  -1.480629482e+00f,
            -1.573856988e+00f,  -1.604608655e+00f,  -1.629279517e+00f,  -1.550780493e+00f,  -1.643634713e+00f,  -1.674171707e+00f,
            -1.483407857e+00f,  -1.404875080e+00f,  -1.498300945e+00f,  -1.529358803e+00f,  -1.555170826e+00f,  -1.476660795e+00f,
            -1.569672808e+00f,  -1.600513602e+00f,  -1.624996418e+00f,  -1.546512800e+00f,  -1.639108585e+00f,  -1.669733564e+00f,
            -1.478332000e+00f,  -1.399838651e+00f,  -1.493006271e+00f,  -1.524123234e+00f,  -1.549714282e+00f,  -1.471264888e+00f,
            -1.563976421e+00f,  -1.594875301e+00f,  -1.619132666e+00f,  -1.540730499e+00f,  -1.632984227e+00f,  -1.663666298e+00f,
            -1.471848954e+00f,  -1.393463004e+00f,  -1.486286016e+00f,  -1.517430894e+00f,  -1.542763512e+00f,  -1.464441761e+00f,
            -1.556767827e+00f,  -1.587693752e+00f,  -1.611688259e+00f,  -1.533433590e+00f,  -1.625261638e+00f,  -1.655969907e+00f,
            7.240569341e-01f,  7.899358977e-01f,  7.035278088e-01f,  6.700327386e-01f,  6.049890026e-01f,  6.709875511e-01f,
            5.849641204e-01f,  5.517669935e-01f,  4.887843143e-01f,  5.548966508e-01f,  4.692614788e-01f,  4.363602584e-01f,
            7.250698327e-01f,  7.911385559e-01f,  7.042647449e-01f,  6.708678259e-01f,  6.061887897e-01f,  6.723450937e-01f,
            5.859200112e-01f,  5.528215282e-01f,  4.902095881e-01f,  5.564483220e-01f,  4.704744910e-01f,  4.376723650e-01f,
            7.250987530e-01f,  7.912496457e-01f,  7.040506448e-01f,  6.707935828e-01f,  6.065344124e-01f,  6.727424672e-01f,
            5.860540139e-01f,  5.530956271e-01f,  4.909094774e-01f,  5.571701761e-01f,  4.709937359e-01f,  4.383319029e-01f,
            7.241436949e-01f,  7.902691669e-01f,  7.028855085e-01f,  6.698100094e-01f,  6.060258708e-01f,  6.721796716e-01f,
            5.853661286e-01f,  5.525892902e-01f,  4.908839823e-01f,  5.570622129e-01f,  4.708192134e-01f,  4.383388721e-01f,
            1.994256417e+00f,  2.041505964e+00f,  1.972811761e+00f,  1.941022683e+00f,  1.854956522e+00f,  1.902522027e+00f,
            1.834029856e+00f,  1.802584294e+00f,  1.719057842e+00f,  1.766930582e+00f,  1.698646800e+00f,  1.667542064e+00f,
            1.995873545e+00f,  2.043455969e+00f,  1.973706318e+00f,  1.942189108e+00f,  1.856627785e+00f,  1.904487601e+00f,
            1.835018349e+00f,  1.803844242e+00f,  1.720828716e+00f,  1.768957993e+00f,  1.699774166e+00f,  1.668940415e+00f,
            1.994218071e+00f,  2.041992164e+00f,  1.971374100e+00f,  1.940178312e+00f,  1.855188081e+00f,  1.903202992e+00f,
            1.832940879e+00f,  1.802087439e+00f,  1.719648866e+00f,  1.767897434e+00f,  1.697994995e+00f,  1.667481098e+00f,
            1.989289994e+00f,  2.037114548e+00f,  1.965815105e+00f,  1.934990293e+00f,  1.850637410e+00f,  1.898668199e+00f,
            1.827797445e+00f,  1.797313884e+00f,  1.715518290e+00f,  1.763748906e+00f,  1.693309287e+00f,  1.663164115e+00f,
            -7.942973518e-01f,  -7.395113163e-01f,  -8.030841512e-01f,  -8.275427105e-01f,  -8.538791613e-01f,  -7.989599366e-01f,
            -8.623009002e-01f,  -8.865496102e-01f,  -9.114216657e-01f,  -8.563765681e-01f,  -9.194787285e-01f,  -9.435192077e-01f,
            -7.770920316e-01f,  -7.224879965e-01f,  -7.857938786e-01f,  -8.101309699e-01f,  -8.360590658e-01f,  -7.813437006e-01f,
            -8.443716642e-01f,  -8.684988541e-01f,  -8.929580047e-01f,  -8.381381738e-01f,  -9.008820547e-01f,  -9.248009966e-01f,
            -7.593895175e-01f,  -7.050510652e-01f,  -7.679825332e-01f,  -7.921623323e-01f,  -8.176360199e-01f,  -7.632069229e-01f,
            -8.258160805e-01f,  -8.497861361e-01f,  -8.737864928e-01f,  -8.192731744e-01f,  -8.815546091e-01f,  -9.053166096e-01f,
            -7.411898097e-01f,  -6.872005224e-01f,  -7.496501149e-01f,  -7.736367978e-01f,  -7.986100236e-01f,  -7.445496036e-01f,
            -8.066341492e-01f,  -8.304114564e-01f,  -8.539071299e-01f,  -7.997815699e-01f,  -8.614963916e-01f,  -8.850660466e-01f,
            1.043181211e+00f,  1.077274350e+00f,  1.028920396e+00f,  1.004901562e+00f,  9.419251550e-01f,  9.763994095e-01f,
            9.281380742e-01f,  9.043989032e-01f,  8.436938662e-01f,  8.785380797e-01f,  8.303793844e-01f,  8.069171856e-01f,
            1.043777995e+00f,  1.077733530e+00f,  1.029016526e+00f,  1.005409842e+00f,  9.431372347e-01f,  9.774428817e-01f,
            9.288859603e-01f,  9.055573761e-01f,  8.455616606e-01f,  8.802069795e-01f,  8.318189338e-01f,  8.087656996e-01f,
            1.041554464e+00f,  1.075239169e+00f,  1.026334298e+00f,  1.003189079e+00f,  9.416895462e-01f,  9.756952802e-01f,
            9.270152523e-01f,  9.041462119e-01f,  8.449287741e-01f,  8.792457964e-01f,  8.307982037e-01f,  8.082025330e-01f,
            1.036510620e+00f,  1.069791267e+00f,  1.020873712e+00f,  9.982392719e-01f,  9.375820893e-01f,  9.711566050e-01f,
            9.225259503e-01f,  9.001654106e-01f,  8.417952064e-01f,  8.756545305e-01f,  8.273171939e-01f,  8.052276857e-01f,
            -9.991099660e-01f,  -9.484296044e-01f,  -1.003150862e+00f,  -1.024593253e+00f,  -1.040658908e+00f,  -9.898930786e-01f,
            -1.044255534e+00f,  -1.065496938e+00f,  -1.080072433e+00f,  -1.029229551e+00f,  -1.083225662e+00f,  -1.104268114e+00f,
            -9.730190794e-01f,  -9.229400110e-01f,  -9.767212949e-01f,  -9.979128800e-01f,  -1.013294605e+00f,  -9.631467486e-01f,
            -1.016529000e+00f,  -1.037520923e+00f,  -1.051408226e+00f,  -1.001199513e+00f,  -1.054176032e+00f,  -1.074970375e+00f,
            -9.455589900e-01f,  -8.961621441e-01f,  -9.489012042e-01f,  -9.698036962e-01f,  -9.844453514e-01f,  -9.349950495e-01f,
            -9.872967339e-01f,  -1.008001161e+00f,  -1.021144428e+00f,  -9.716481305e-01f,  -1.023506560e+00f,  -1.044015049e+00f,
            -9.167296978e-01f,  -8.680960036e-01f,  -9.196905894e-01f,  -9.402657015e-01f,  -9.541111482e-01f,  -9.054379813e-01f,
            -9.565587350e-01f,  -9.769376509e-01f,  -9.892810380e-01f,  -9.405754021e-01f,  -9.912172459e-01f,  -1.011402138e+00f,
            -1.688048810e+00f,  -1.635515962e+00f,  -1.684045964e+00f,  -1.701949262e+00f,  -1.689856821e+00f,  -1.637389599e+00f,
            -1.685502238e+00f,  -1.703276878e+00f,  -1.690359742e+00f,  -1.637964883e+00f,  -1.685653100e+00f,  -1.703300304e+00f,
            -1.635852820e+00f,  -1.584019770e+00f,  -1.631139240e+00f,  -1.648914405e+00f,  -1.636277810e+00f,  -1.584519966e+00f,
            -1.631197673e+00f,  -1.648844620e+00f,  -1.635380128e+00f,  -1.583703952e+00f,  -1.629933318e+00f,  -1.647453298e+00f,
            -1.580214072e+00f,  -1.529129204e+00f,  -1.574780082e+00f,  -1.592398083e+00f,  -1.579177937e+00f,  -1.528177061e+00f,
            -1.573362883e+00f,  -1.590853273e+00f,  -1.576802225e+00f,  -1.525891539e+00f,  -1.570606195e+00f,  -1.587970253e+00f,
            -1.521132565e+00f,  -1.470844264e+00f,  -1.514968490e+00f,  -1.532400294e+00f,  -1.518557202e+00f,  -1.468360885e+00f,
            -1.511997868e+00f,  -1.529302835e+00f,  -1.514626034e+00f,  -1.464527641e+00f,  -1.507671731e+00f,  -1.524851167e+00f,
            -8.648460453e-01f,  -8.255452719e-01f,  -8.628423259e-01f,  -8.787918921e-01f,  -8.813104679e-01f,  -8.419337335e-01f,
            -8.788665494e-01f,  -8.946637598e-01f,  -8.959256080e-01f,  -8.564824223e-01f,  -8.930417850e-01f,  -9.086886677e-01f,
            -8.237074938e-01f,  -7.852592622e-01f,  -8.212426329e-01f,  -8.368343589e-01f,  -8.384898538e-01f,  -7.999773987e-01f,
            -8.355634397e-01f,  -8.510051053e-01f,  -8.514009548e-01f,  -8.128333575e-01f,  -8.480136177e-01f,  -8.633072981e-01f,
            -7.807986928e-01f,  -7.432784957e-01f,  -7.778538902e-01f,  -7.930498834e-01f,  -7.937849211e-01f,  -7.562109415e-01f,
            -7.903577181e-01f,  -8.054062295e-01f,  -8.048790677e-01f,  -7.672599715e-01f,  -8.009704229e-01f,  -8.158735753e-01f,
            -7.361196421e-01f,  -6.996029724e-01f,  -7.326760980e-01f,  -7.474384655e-01f,  -7.471956698e-01f,  -7.106343619e-01f,
            -7.432493845e-01f,  -7.578671325e-01f,  -7.563599468e-01f,  -7.197622644e-01f,  -7.519122009e-01f,  -7.663874993e-01f,
            -1.252176468e+00f,  -1.205957685e+00f,  -1.241406644e+00f,  -1.256405542e+00f,  -1.234223320e+00f,  -1.188179776e+00f,
            -1.223061739e+00f,  -1.237932959e+00f,  -1.215013074e+00f,  -1.169152253e+00f,  -1.203461074e+00f,  -1.218206653e+00f,
            -1.171600396e+00f,  -1.126292043e+00f,  -1.159913881e+00f,  -1.174723621e+00f,  -1.151826944e+00f,  -1.106697279e+00f,
            -1.139735914e+00f,  -1.154420931e+00f,  -1.130789811e+00f,  -1.085846073e+00f,  -1.118295844e+00f,  -1.132858218e+00f,
            -1.086534807e+00f,  -1.042177630e+00f,  -1.073925484e+00f,  -1.088517553e+00f,  -1.064865112e+00f,  -1.020689202e+00f,
            -1.051838817e+00f,  -1.066309388e+00f,  -1.041926261e+00f,  -9.979386238e-01f,  -1.028484801e+00f,  -1.042836002e+00f,
            -9.969797002e-01f,  -9.536144464e-01f,  -9.834414537e-01f,  -9.977873367e-01f,  -9.733378259e-01f,  -9.301555459e-01f,
            -9.593704480e-01f,  -9.735983310e-01f,  -9.484224238e-01f,  -9.054299063e-01f,  -9.340279479e-01f,  -9.481400031e-01f,
            -9.878865236e-01f,  -9.439152907e-01f,  -9.717930489e-01f,  -9.853439782e-01f,  -9.539351938e-01f,  -9.102324190e-01f,
            -9.374982476e-01f,  -9.509645295e-01f,  -9.192057220e-01f,  -8.757773544e-01f,  -9.024253765e-01f,  -9.158082335e-01f,
            -8.820414035e-01f,  -8.389368351e-01f,  -8.648884823e-01f,  -8.783300425e-01f,  -8.463459137e-01f,  -8.035120391e-01f,
            -8.288409322e-01f,  -8.421992845e-01f,  -8.098669000e-01f,  -7.673095381e-01f,  -7.920100667e-01f,  -8.052864613e-01f,
            -7.707187264e-01f,  -7.285056559e-01f,  -7.525054615e-01f,  -7.658146138e-01f,  -7.332253740e-01f,  -6.912847263e-01f,
            -7.146516301e-01f,  -7.278791805e-01f,  -6.949437938e-01f,  -6.532812653e-01f,  -6.760099116e-01f,  -6.891571381e-01f,
            -6.539184924e-01f,  -6.126217533e-01f,  -6.346439863e-01f,  -6.477976920e-01f,  -6.145735748e-01f,  -5.735504806e-01f,
            -5.949303414e-01f,  -6.080042175e-01f,  -5.744364033e-01f,  -5.336925359e-01f,  -5.544249114e-01f,  -5.674202638e-01f,
            -4.422518848e-01f,  -4.076710776e-01f,  -4.256069968e-01f,  -4.368585888e-01f,  -4.094322700e-01f,  -3.750622510e-01f,
            -3.923855940e-01f,  -4.035578413e-01f,  -3.756369620e-01f,  -3.414855449e-01f,  -3.581896409e-01f,  -3.692845716e-01f,
            -3.363289560e-01f,  -3.027665922e-01f,  -3.187097653e-01f,  -3.297013286e-01f,  -3.014578304e-01f,  -2.681067437e-01f,
            -2.834254779e-01f,  -2.943416925e-01f,  -2.656097999e-01f,  -2.324776143e-01f,  -2.471656443e-01f,  -2.580085853e-01f,
            -2.256059743e-01f,  -1.930992276e-01f,  -2.070077389e-01f,  -2.177115839e-01f,  -1.886107981e-01f,  -1.563149984e-01f,
            -1.695882948e-01f,  -1.802210703e-01f,  -1.506386587e-01f,  -1.185612537e-01f,  -1.311934554e-01f,  -1.417572840e-01f,
            -1.100829397e-01f,  -7.866898376e-02f,  -9.050091754e-02f,  -1.008893547e-01f,  -7.089117293e-02f,  -3.968701512e-02f,
            -5.087404458e-02f,  -6.119597495e-02f,  -3.072353844e-02f,  2.635368739e-04f,  -1.027307429e-02f,  -2.053066763e-02f,
        };
    }
}
