using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection1D {
    [TestClass]
    public class DeconvolutionTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, kwidth = 3, inwidth = 13;
            int outwidth = inwidth - kwidth + 1, batch = 2;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(inchannels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(outchannels, outwidth, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel1D(inchannels, outchannels, kwidth), wval);

            VariableField x_actual = xtensor;
            ParameterField w = wtensor;
            ParameterField y = ytensor;

            Field x_expect = Deconvolution1D(y, w);
            Field err = Abs(x_expect - x_actual);
            StoreField err_store = err;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] err_actual = err_store.State;
            float[] gy_actual = y.GradTensor.State;
            float[] gw_actual = w.GradTensor.State;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gy_expect, gy_actual, 1e-7f, 1e-5f, $"not equal gy");
        }

        float[] gy_expect = new float[] {
            7.158443600e-02f, 6.690929700e-02f, 6.223415800e-02f, 5.755901900e-02f, 5.288388000e-02f, 4.820874100e-02f, 4.353360200e-02f, 3.885846300e-02f, 3.418332400e-02f, 2.950818500e-02f, 2.483304600e-02f,
            1.570224320e-01f, 1.480353420e-01f, 1.390482520e-01f, 1.300611620e-01f, 1.210740720e-01f, 1.120869820e-01f, 1.030998920e-01f, 9.411280200e-02f, 8.512571200e-02f, 7.613862200e-02f, 6.715153200e-02f,
            2.612890840e-01f, 2.473551010e-01f, 2.334211180e-01f, 2.194871350e-01f, 2.055531520e-01f, 1.916191690e-01f, 1.776851860e-01f, 1.637512030e-01f, 1.498172200e-01f, 1.358832370e-01f, 1.219492540e-01f,
            3.707663260e-01f, 3.517248280e-01f, 3.326833300e-01f, 3.136418320e-01f, 2.946003340e-01f, 2.755588360e-01f, 2.565173380e-01f, 2.374758400e-01f, 2.184343420e-01f, 1.993928440e-01f, 1.803513460e-01f,
            4.802435680e-01f, 4.560945550e-01f, 4.319455420e-01f, 4.077965290e-01f, 3.836475160e-01f, 3.594985030e-01f, 3.353494900e-01f, 3.112004770e-01f, 2.870514640e-01f, 2.629024510e-01f, 2.387534380e-01f,
            5.897208100e-01f, 5.604642820e-01f, 5.312077540e-01f, 5.019512260e-01f, 4.726946980e-01f, 4.434381700e-01f, 4.141816420e-01f, 3.849251140e-01f, 3.556685860e-01f, 3.264120580e-01f, 2.971555300e-01f,
            6.991980520e-01f, 6.648340090e-01f, 6.304699660e-01f, 5.961059230e-01f, 5.617418800e-01f, 5.273778370e-01f, 4.930137940e-01f, 4.586497510e-01f, 4.242857080e-01f, 3.899216650e-01f, 3.555576220e-01f,
            8.086752940e-01f, 7.692037360e-01f, 7.297321780e-01f, 6.902606200e-01f, 6.507890620e-01f, 6.113175040e-01f, 5.718459460e-01f, 5.323743880e-01f, 4.929028300e-01f, 4.534312720e-01f, 4.139597140e-01f,
            9.181525360e-01f, 8.735734630e-01f, 8.289943900e-01f, 7.844153170e-01f, 7.398362440e-01f, 6.952571710e-01f, 6.506780980e-01f, 6.060990250e-01f, 5.615199520e-01f, 5.169408790e-01f, 4.723618060e-01f,
            8.920012080e-01f, 8.553163780e-01f, 8.186315480e-01f, 7.819467180e-01f, 7.452618880e-01f, 7.085770580e-01f, 6.718922280e-01f, 6.352073980e-01f, 5.985225680e-01f, 5.618377380e-01f, 5.251529080e-01f,
            6.298619740e-01f, 6.099821350e-01f, 5.901022960e-01f, 5.702224570e-01f, 5.503426180e-01f, 5.304627790e-01f, 5.105829400e-01f, 4.907031010e-01f, 4.708232620e-01f, 4.509434230e-01f, 4.310635840e-01f,
            8.549266670e-01f, 8.085836920e-01f, 7.622407170e-01f, 7.158977420e-01f, 6.695547670e-01f, 6.232117920e-01f, 5.768688170e-01f, 5.305258420e-01f, 4.841828670e-01f, 4.378398920e-01f, 3.914969170e-01f,
            1.236766384e+00f, 1.176132951e+00f, 1.115499518e+00f, 1.054866085e+00f, 9.942326520e-01f, 9.335992190e-01f, 8.729657860e-01f, 8.123323530e-01f, 7.516989200e-01f, 6.910654870e-01f, 6.304320540e-01f,
            1.421438746e+00f, 1.353380098e+00f, 1.285321450e+00f, 1.217262802e+00f, 1.149204154e+00f, 1.081145506e+00f, 1.013086858e+00f, 9.450282100e-01f, 8.769695620e-01f, 8.089109140e-01f, 7.408522660e-01f,
            1.530915988e+00f, 1.457749825e+00f, 1.384583662e+00f, 1.311417499e+00f, 1.238251336e+00f, 1.165085173e+00f, 1.091919010e+00f, 1.018752847e+00f, 9.455866840e-01f, 8.724205210e-01f, 7.992543580e-01f,
            1.640393230e+00f, 1.562119552e+00f, 1.483845874e+00f, 1.405572196e+00f, 1.327298518e+00f, 1.249024840e+00f, 1.170751162e+00f, 1.092477484e+00f, 1.014203806e+00f, 9.359301280e-01f, 8.576564500e-01f,
            1.749870472e+00f, 1.666489279e+00f, 1.583108086e+00f, 1.499726893e+00f, 1.416345700e+00f, 1.332964507e+00f, 1.249583314e+00f, 1.166202121e+00f, 1.082820928e+00f, 9.994397350e-01f, 9.160585420e-01f,
            1.859347714e+00f, 1.770859006e+00f, 1.682370298e+00f, 1.593881590e+00f, 1.505392882e+00f, 1.416904174e+00f, 1.328415466e+00f, 1.239926758e+00f, 1.151438050e+00f, 1.062949342e+00f, 9.744606340e-01f,
            1.968824956e+00f, 1.875228733e+00f, 1.781632510e+00f, 1.688036287e+00f, 1.594440064e+00f, 1.500843841e+00f, 1.407247618e+00f, 1.313651395e+00f, 1.220055172e+00f, 1.126458949e+00f, 1.032862726e+00f,
            2.078302198e+00f, 1.979598460e+00f, 1.880894722e+00f, 1.782190984e+00f, 1.683487246e+00f, 1.584783508e+00f, 1.486079770e+00f, 1.387376032e+00f, 1.288672294e+00f, 1.189968556e+00f, 1.091264818e+00f,
            1.921526530e+00f, 1.843239083e+00f, 1.764951636e+00f, 1.686664189e+00f, 1.608376742e+00f, 1.530089295e+00f, 1.451801848e+00f, 1.373514401e+00f, 1.295226954e+00f, 1.216939507e+00f, 1.138652060e+00f,
            1.312766945e+00f, 1.271306722e+00f, 1.229846499e+00f, 1.188386276e+00f, 1.146926053e+00f, 1.105465830e+00f, 1.064005607e+00f, 1.022545384e+00f, 9.810851610e-01f, 9.396249380e-01f, 8.981647150e-01f,
        };

        float[] gw_expect = new float[] {
            1.229860456e+00f, 1.215059494e+00f, 1.200258532e+00f, 1.185457570e+00f, 1.170656608e+00f, 1.155855646e+00f, 1.141054684e+00f,
            1.237603972e+00f, 1.222704868e+00f, 1.207805764e+00f, 1.192906660e+00f, 1.178007556e+00f, 1.163108452e+00f, 1.148209348e+00f,
            1.245347488e+00f, 1.230350242e+00f, 1.215352996e+00f, 1.200355750e+00f, 1.185358504e+00f, 1.170361258e+00f, 1.155364012e+00f,
            1.253091004e+00f, 1.237995616e+00f, 1.222900228e+00f, 1.207804840e+00f, 1.192709452e+00f, 1.177614064e+00f, 1.162518676e+00f,
            1.260834520e+00f, 1.245640990e+00f, 1.230447460e+00f, 1.215253930e+00f, 1.200060400e+00f, 1.184866870e+00f, 1.169673340e+00f,
            1.268578036e+00f, 1.253286364e+00f, 1.237994692e+00f, 1.222703020e+00f, 1.207411348e+00f, 1.192119676e+00f, 1.176828004e+00f,
            1.276321552e+00f, 1.260931738e+00f, 1.245541924e+00f, 1.230152110e+00f, 1.214762296e+00f, 1.199372482e+00f, 1.183982668e+00f,
            1.284065068e+00f, 1.268577112e+00f, 1.253089156e+00f, 1.237601200e+00f, 1.222113244e+00f, 1.206625288e+00f, 1.191137332e+00f,
            1.291808584e+00f, 1.276222486e+00f, 1.260636388e+00f, 1.245050290e+00f, 1.229464192e+00f, 1.213878094e+00f, 1.198291996e+00f,
            1.299552100e+00f, 1.283867860e+00f, 1.268183620e+00f, 1.252499380e+00f, 1.236815140e+00f, 1.221130900e+00f, 1.205446660e+00f,
            1.307295616e+00f, 1.291513234e+00f, 1.275730852e+00f, 1.259948470e+00f, 1.244166088e+00f, 1.228383706e+00f, 1.212601324e+00f,
            1.193134778e+00f, 1.177886842e+00f, 1.162638906e+00f, 1.147390970e+00f, 1.132143034e+00f, 1.116895098e+00f, 1.101647162e+00f,
            1.201045736e+00f, 1.185693619e+00f, 1.170341502e+00f, 1.154989385e+00f, 1.139637268e+00f, 1.124285151e+00f, 1.108933034e+00f,
            1.208956694e+00f, 1.193500396e+00f, 1.178044098e+00f, 1.162587800e+00f, 1.147131502e+00f, 1.131675204e+00f, 1.116218906e+00f,
            1.216867652e+00f, 1.201307173e+00f, 1.185746694e+00f, 1.170186215e+00f, 1.154625736e+00f, 1.139065257e+00f, 1.123504778e+00f,
            1.224778610e+00f, 1.209113950e+00f, 1.193449290e+00f, 1.177784630e+00f, 1.162119970e+00f, 1.146455310e+00f, 1.130790650e+00f,
            1.232689568e+00f, 1.216920727e+00f, 1.201151886e+00f, 1.185383045e+00f, 1.169614204e+00f, 1.153845363e+00f, 1.138076522e+00f,
            1.240600526e+00f, 1.224727504e+00f, 1.208854482e+00f, 1.192981460e+00f, 1.177108438e+00f, 1.161235416e+00f, 1.145362394e+00f,
            1.248511484e+00f, 1.232534281e+00f, 1.216557078e+00f, 1.200579875e+00f, 1.184602672e+00f, 1.168625469e+00f, 1.152648266e+00f,
            1.256422442e+00f, 1.240341058e+00f, 1.224259674e+00f, 1.208178290e+00f, 1.192096906e+00f, 1.176015522e+00f, 1.159934138e+00f,
            1.264333400e+00f, 1.248147835e+00f, 1.231962270e+00f, 1.215776705e+00f, 1.199591140e+00f, 1.183405575e+00f, 1.167220010e+00f,
            1.272244358e+00f, 1.255954612e+00f, 1.239664866e+00f, 1.223375120e+00f, 1.207085374e+00f, 1.190795628e+00f, 1.174505882e+00f,
            1.047467476e+00f, 1.032700394e+00f, 1.017933312e+00f, 1.003166230e+00f, 9.883991480e-01f, 9.736320660e-01f, 9.588649840e-01f,
            1.054886052e+00f, 1.040014052e+00f, 1.025142052e+00f, 1.010270052e+00f, 9.953980520e-01f, 9.805260520e-01f, 9.656540520e-01f,
            1.062304628e+00f, 1.047327710e+00f, 1.032350792e+00f, 1.017373874e+00f, 1.002396956e+00f, 9.874200380e-01f, 9.724431200e-01f,
            1.069723204e+00f, 1.054641368e+00f, 1.039559532e+00f, 1.024477696e+00f, 1.009395860e+00f, 9.943140240e-01f, 9.792321880e-01f,
            1.077141780e+00f, 1.061955026e+00f, 1.046768272e+00f, 1.031581518e+00f, 1.016394764e+00f, 1.001208010e+00f, 9.860212560e-01f,
            1.084560356e+00f, 1.069268684e+00f, 1.053977012e+00f, 1.038685340e+00f, 1.023393668e+00f, 1.008101996e+00f, 9.928103240e-01f,
            1.091978932e+00f, 1.076582342e+00f, 1.061185752e+00f, 1.045789162e+00f, 1.030392572e+00f, 1.014995982e+00f, 9.995993920e-01f,
            1.099397508e+00f, 1.083896000e+00f, 1.068394492e+00f, 1.052892984e+00f, 1.037391476e+00f, 1.021889968e+00f, 1.006388460e+00f,
            1.106816084e+00f, 1.091209658e+00f, 1.075603232e+00f, 1.059996806e+00f, 1.044390380e+00f, 1.028783954e+00f, 1.013177528e+00f,
            1.114234660e+00f, 1.098523316e+00f, 1.082811972e+00f, 1.067100628e+00f, 1.051389284e+00f, 1.035677940e+00f, 1.019966596e+00f,
            1.121653236e+00f, 1.105836974e+00f, 1.090020712e+00f, 1.074204450e+00f, 1.058388188e+00f, 1.042571926e+00f, 1.026755664e+00f,
        };
    }
}
