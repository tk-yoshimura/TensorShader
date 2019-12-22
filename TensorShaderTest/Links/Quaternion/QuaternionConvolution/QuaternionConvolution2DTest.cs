using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.QuaternionConvolution {
    [TestClass]
    public class QuaternionConvolution2DTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 8, outchannels = 12, kwidth = 3, kheight = 5, inwidth = 7, inheight = 8;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, batch = 3;

            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * outchannels * inchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel2D(inchannels, outchannels / 4, kwidth, kheight), wval);

            ParameterField x = xtensor;
            ParameterField w = wtensor;
            VariableField y_actual = ytensor;

            Field y_expect = QuaternionConvolution2D(x, w);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;
            float[] gw_actual = w.GradTensor.State;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int inchannels = 8, outchannels = 12, kwidth = 3, kheight = 5, inwidth = 7, inheight = 8;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, batch = 3;

            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * outchannels * inchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map2D(outchannels, outwidth, outheight, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel2D(inchannels, outchannels / 4, kwidth, kheight), wval);

            ParameterField x = xtensor;
            ParameterField w = wtensor;
            VariableField y_actual = ytensor;

            Field xr = QuaternionR(x), xi = QuaternionI(x), xj = QuaternionJ(x), xk = QuaternionK(x);
            Field wr = QuaternionR(w), wi = QuaternionI(w), wj = QuaternionJ(w), wk = QuaternionK(w);

            Field yr = Convolution2D(xr, wr) - Convolution2D(xi, wi) - Convolution2D(xj, wj) - Convolution2D(xk, wk);
            Field yi = Convolution2D(xr, wi) + Convolution2D(xi, wr) + Convolution2D(xj, wk) - Convolution2D(xk, wj);
            Field yj = Convolution2D(xr, wj) - Convolution2D(xi, wk) + Convolution2D(xj, wr) + Convolution2D(xk, wi);
            Field yk = Convolution2D(xr, wk) + Convolution2D(xi, wj) - Convolution2D(xj, wi) + Convolution2D(xk, wr);

            Field y_expect = QuaternionCast(yr, yi, yj, yk);

            Field err = y_expect - y_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;
            float[] gw_actual = w.GradTensor.State;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = new float[] {
            1.719083920e+00f,  1.730762940e+00f,  1.776419160e+00f,  1.784434500e+00f,  1.699347040e+00f,  1.711062540e+00f, 
            1.756082760e+00f,  1.764061140e+00f,  3.450956960e+00f,  3.523321080e+00f,  3.611868720e+00f,  3.627044040e+00f, 
            3.409974080e+00f,  3.481869720e+00f,  3.569122320e+00f,  3.584235240e+00f,  5.186564400e+00f,  5.365371060e+00f, 
            5.493907080e+00f,  5.515456140e+00f,  5.122826400e+00f,  5.300118180e+00f,  5.426677080e+00f,  5.448149820e+00f, 
            5.553031440e+00f,  5.868453780e+00f,  5.999728680e+00f,  6.019584300e+00f,  5.484766080e+00f,  5.797049220e+00f, 
            5.926277880e+00f,  5.946091740e+00f,  5.919498480e+00f,  6.371536500e+00f,  6.505550280e+00f,  6.523712460e+00f, 
            5.846705760e+00f,  6.293980260e+00f,  6.425878680e+00f,  6.444033660e+00f,  3.910829600e+00f,  4.243868280e+00f, 
            4.329708720e+00f,  4.341330120e+00f,  3.860792000e+00f,  4.190113560e+00f,  4.274520720e+00f,  4.286148840e+00f, 
            1.934645200e+00f,  2.115921660e+00f,  2.157044760e+00f,  2.162635140e+00f,  1.908871840e+00f,  2.088019020e+00f, 
            2.128413960e+00f,  2.134013460e+00f,  4.102032800e+00f,  4.367045880e+00f,  4.454124720e+00f,  4.465342920e+00f, 
            4.050843200e+00f,  4.313291160e+00f,  4.398936720e+00f,  4.410161640e+00f,  8.132184640e+00f,  8.734648560e+00f, 
            8.901894240e+00f,  8.923311120e+00f,  8.026787200e+00f,  8.623038000e+00f,  8.787371040e+00f,  8.808824400e+00f, 
            1.207234608e+01f,  1.307820132e+01f,  1.331842536e+01f,  1.334915964e+01f,  1.190972256e+01f,  1.290463380e+01f, 
            1.314041976e+01f,  1.317124332e+01f,  1.272378768e+01f,  1.397363652e+01f,  1.421809416e+01f,  1.424606364e+01f, 
            1.255210944e+01f,  1.378776564e+01f,  1.402764696e+01f,  1.405577484e+01f,  1.337522928e+01f,  1.486907172e+01f, 
            1.511776296e+01f,  1.514296764e+01f,  1.319449632e+01f,  1.467089748e+01f,  1.491487416e+01f,  1.494030636e+01f, 
            8.748354880e+00f,  9.782035440e+00f,  9.939443040e+00f,  9.955963920e+00f,  8.624848000e+00f,  9.645818160e+00f, 
            9.800036640e+00f,  9.816732240e+00f,  4.283908640e+00f,  4.817475960e+00f,  4.891894320e+00f,  4.900059720e+00f, 
            4.220646080e+00f,  4.747316760e+00f,  4.820117520e+00f,  4.828381800e+00f,  6.937961520e+00f,  7.650478260e+00f, 
            7.771843080e+00f,  7.782903180e+00f,  6.843603360e+00f,  7.548315300e+00f,  7.667288280e+00f,  7.678479420e+00f, 
            1.362191280e+01f,  1.511724132e+01f,  1.534752936e+01f,  1.536915708e+01f,  1.342866912e+01f,  1.490676372e+01f, 
            1.513219896e+01f,  1.515412332e+01f,  2.002468968e+01f,  2.236337910e+01f,  2.268973404e+01f,  2.272164426e+01f, 
            1.972803312e+01f,  2.203843518e+01f,  2.235740724e+01f,  2.238981426e+01f,  2.087961336e+01f,  2.354043654e+01f, 
            2.387127564e+01f,  2.389997178e+01f,  2.056937472e+01f,  2.319703758e+01f,  2.352028644e+01f,  2.354958306e+01f, 
            2.173453704e+01f,  2.471749398e+01f,  2.505281724e+01f,  2.507829930e+01f,  2.141071632e+01f,  2.435563998e+01f, 
            2.468316564e+01f,  2.470935186e+01f,  1.409080560e+01f,  1.609776036e+01f,  1.630665576e+01f,  1.632425724e+01f, 
            1.387039776e+01f,  1.585037268e+01f,  1.605400056e+01f,  1.607210604e+01f,  6.836905200e+00f,  7.846292340e+00f, 
            7.943275080e+00f,  7.952451660e+00f,  6.724437600e+00f,  7.719522660e+00f,  7.813837080e+00f,  7.823282940e+00f, 
            1.001598496e+01f,  1.132268952e+01f,  1.146830064e+01f,  1.147729320e+01f,  9.866742400e+00f,  1.115776440e+01f, 
            1.129986384e+01f,  1.130919240e+01f,  1.949837120e+01f,  2.215435824e+01f,  2.242622688e+01f,  2.244493776e+01f, 
            1.919384960e+01f,  2.181630576e+01f,  2.208105888e+01f,  2.210048784e+01f,  2.841093984e+01f,  3.244579272e+01f, 
            3.282401232e+01f,  3.285344376e+01f,  2.794510272e+01f,  3.192641064e+01f,  3.229381872e+01f,  3.232439640e+01f, 
            2.938785312e+01f,  3.379374216e+01f,  3.417545232e+01f,  3.420184248e+01f,  2.890390656e+01f,  3.324975336e+01f, 
            3.362037552e+01f,  3.364805016e+01f,  3.036476640e+01f,  3.514169160e+01f,  3.552689232e+01f,  3.555024120e+01f, 
            2.986271040e+01f,  3.457309608e+01f,  3.494693232e+01f,  3.497170392e+01f,  1.951641152e+01f,  2.267430192e+01f, 
            2.290879968e+01f,  2.292656592e+01f,  1.917567104e+01f,  2.228703600e+01f,  2.251386528e+01f,  2.253262608e+01f, 
            9.382749760e+00f,  1.094400024e+01f,  1.104991344e+01f,  1.105998888e+01f,  9.209361280e+00f,  1.074626616e+01f, 
            1.084829904e+01f,  1.085889480e+01f,  7.329618880e+00f,  8.354037360e+00f,  8.436438240e+00f,  8.451478800e+00f, 
            7.180376320e+00f,  8.189112240e+00f,  8.268001440e+00f,  8.283378000e+00f,  1.401698240e+01f,  1.606941360e+01f, 
            1.621320288e+01f,  1.624483920e+01f,  1.371246080e+01f,  1.573136112e+01f,  1.586803488e+01f,  1.590038928e+01f, 
            2.002587168e+01f,  2.309691528e+01f,  2.328052752e+01f,  2.333059128e+01f,  1.956003456e+01f,  2.257753320e+01f, 
            2.275033392e+01f,  2.280154392e+01f,  2.067681504e+01f,  2.400194376e+01f,  2.418406992e+01f,  2.423358072e+01f, 
            2.019286848e+01f,  2.345795496e+01f,  2.362899312e+01f,  2.367978840e+01f,  2.132775840e+01f,  2.490697224e+01f, 
            2.508761232e+01f,  2.513657016e+01f,  2.082570240e+01f,  2.433837672e+01f,  2.450765232e+01f,  2.455803288e+01f, 
            1.338308288e+01f,  1.570351536e+01f,  1.579998048e+01f,  1.583564880e+01f,  1.304234240e+01f,  1.531624944e+01f, 
            1.540504608e+01f,  1.544170896e+01f,  6.261757120e+00f,  7.384786800e+00f,  7.420854240e+00f,  7.440295440e+00f, 
            6.088368640e+00f,  7.187052720e+00f,  7.219239840e+00f,  7.239201360e+00f,  4.700488080e+00f,  5.427658260e+00f, 
            5.458432680e+00f,  5.474832300e+00f,  4.570982400e+00f,  5.282433540e+00f,  5.310332280e+00f,  5.327104860e+00f, 
            8.773095840e+00f,  1.019177028e+01f,  1.023548616e+01f,  1.027048860e+01f,  8.509557120e+00f,  9.895169160e+00f, 
            9.933064560e+00f,  9.968847480e+00f,  1.219065912e+01f,  1.425542598e+01f,  1.429383564e+01f,  1.434985146e+01f, 
            1.178856000e+01f,  1.380129678e+01f,  1.383087204e+01f,  1.388811042e+01f,  1.255662792e+01f,  1.476810198e+01f, 
            1.480353084e+01f,  1.486006506e+01f,  1.214094672e+01f,  1.429551774e+01f,  1.432190484e+01f,  1.437976530e+01f, 
            1.292259672e+01f,  1.528077798e+01f,  1.531322604e+01f,  1.537027866e+01f,  1.249333344e+01f,  1.478973870e+01f, 
            1.481293764e+01f,  1.487142018e+01f,  7.842308640e+00f,  9.326785320e+00f,  9.328372560e+00f,  9.369716760e+00f, 
            7.551605760e+00f,  8.993274120e+00f,  8.988626160e+00f,  9.030958200e+00f,  3.525721680e+00f,  4.220889300e+00f, 
            4.211522280e+00f,  4.233918060e+00f,  3.378106560e+00f,  4.051057860e+00f,  4.038538680e+00f,  4.061445660e+00f, 
            2.461917920e+00f,  2.887611960e+00f,  2.885508720e+00f,  2.900822280e+00f,  2.363864960e+00f,  2.776441560e+00f, 
            2.772259920e+00f,  2.787902760e+00f,  4.407789760e+00f,  5.209826160e+00f,  5.192348640e+00f,  5.225135760e+00f, 
            4.208665600e+00f,  4.983384240e+00f,  4.961703840e+00f,  4.995172560e+00f,  5.819506080e+00f,  6.942035880e+00f, 
            6.895636560e+00f,  6.948195480e+00f,  5.516292480e+00f,  6.596221320e+00f,  6.543448560e+00f,  6.597064440e+00f, 
            5.981992800e+00f,  7.173089640e+00f,  7.123458960e+00f,  7.176985560e+00f,  5.669724480e+00f,  6.814971720e+00f, 
            6.758829360e+00f,  6.813482040e+00f,  6.144479520e+00f,  7.404143400e+00f,  7.351281360e+00f,  7.405775640e+00f, 
            5.823156480e+00f,  7.033722120e+00f,  6.974210160e+00f,  7.029899640e+00f,  3.483689920e+00f,  4.239462000e+00f, 
            4.189475040e+00f,  4.228701840e+00f,  3.266456320e+00f,  3.988413360e+00f,  3.933947040e+00f,  3.973993680e+00f, 
            1.429493600e+00f,  1.763211960e+00f,  1.730753520e+00f,  1.751861640e+00f,  1.319367680e+00f,  1.635637080e+00f, 
            1.600915920e+00f,  1.622445480e+00f,  8.247935200e-01f,  9.922690200e-01f,  9.789399600e-01f,  9.892708200e-01f, 
            7.699091200e-01f,  9.295068600e-01f,  9.150579600e-01f,  9.255937800e-01f,  1.342834400e+00f,  1.640322360e+00f, 
            1.606337520e+00f,  1.628424840e+00f,  1.231556480e+00f,  1.512747480e+00f,  1.476499920e+00f,  1.499008680e+00f, 
            1.545067920e+00f,  1.931856660e+00f,  1.869751080e+00f,  1.905089580e+00f,  1.375887360e+00f,  1.737418500e+00f, 
            1.671884280e+00f,  1.707872220e+00f,  1.585565040e+00f,  1.992018420e+00f,  1.927675080e+00f,  1.963808460e+00f, 
            1.411857120e+00f,  1.791428580e+00f,  1.723587480e+00f,  1.760404860e+00f,  1.626062160e+00f,  2.052180180e+00f, 
            1.985599080e+00f,  2.022527340e+00f,  1.447826880e+00f,  1.845438660e+00f,  1.775290680e+00f,  1.812937500e+00f, 
            7.289969600e-01f,  9.582865200e-01f,  9.058351200e-01f,  9.322482000e-01f,  6.086643200e-01f,  8.184082800e-01f, 
            7.635559200e-01f,  7.904595600e-01f,  1.839580000e-01f,  2.701253400e-01f,  2.398215600e-01f,  2.539482600e-01f, 
            1.230371200e-01f,  1.991609400e-01f,  1.676451600e-01f,  1.820229000e-01f,  1.087867432e+01f,  1.181322054e+01f, 
            1.191515196e+01f,  1.189286970e+01f,  1.075369072e+01f,  1.167868878e+01f,  1.177869396e+01f,  1.175701986e+01f, 
            2.113865744e+01f,  2.299924812e+01f,  2.319260472e+01f,  2.315105556e+01f,  2.088718112e+01f,  2.272813404e+01f, 
            2.291761512e+01f,  2.287729380e+01f,  3.077089464e+01f,  3.354577938e+01f,  3.381991668e+01f,  3.376218510e+01f, 
            3.039141648e+01f,  3.313603242e+01f,  3.340432188e+01f,  3.334844934e+01f,  3.113736168e+01f,  3.404886210e+01f, 
            3.432573828e+01f,  3.426631326e+01f,  3.075335616e+01f,  3.363296346e+01f,  3.390392268e+01f,  3.384639126e+01f, 
            3.150382872e+01f,  3.455194482e+01f,  3.483155988e+01f,  3.477044142e+01f,  3.111529584e+01f,  3.412989450e+01f, 
            3.440352348e+01f,  3.434433318e+01f,  2.033556944e+01f,  2.234181900e+01f,  2.251698552e+01f,  2.247962388e+01f, 
            2.007503840e+01f,  2.205840156e+01f,  2.222955432e+01f,  2.219348964e+01f,  9.831274960e+00f,  1.082040294e+01f, 
            1.090231836e+01f,  1.088535258e+01f,  9.700254880e+00f,  1.067766894e+01f,  1.075756596e+01f,  1.074125442e+01f, 
            2.052677264e+01f,  2.246499660e+01f,  2.264140152e+01f,  2.260363668e+01f,  2.026508960e+01f,  2.218157916e+01f, 
            2.235397032e+01f,  2.231750244e+01f,  3.971870368e+01f,  4.355257368e+01f,  4.388298864e+01f,  4.381418088e+01f, 
            3.919231936e+01f,  4.298163768e+01f,  4.330397904e+01f,  4.323778824e+01f,  5.755768368e+01f,  6.323812452e+01f, 
            6.369987816e+01f,  6.360688764e+01f,  5.676357984e+01f,  6.237556884e+01f,  6.282514296e+01f,  6.273611244e+01f, 
            5.820912528e+01f,  6.413355972e+01f,  6.459954696e+01f,  6.450379164e+01f,  5.740596672e+01f,  6.325870068e+01f, 
            6.371237016e+01f,  6.362064396e+01f,  5.886056688e+01f,  6.502899492e+01f,  6.549921576e+01f,  6.540069564e+01f, 
            5.804835360e+01f,  6.414183252e+01f,  6.459959736e+01f,  6.450517548e+01f,  3.780895264e+01f,  4.184400792e+01f, 
            4.213361904e+01f,  4.207539816e+01f,  3.726445888e+01f,  4.124846520e+01f,  4.152972624e+01f,  4.147426056e+01f, 
            1.818272720e+01f,  2.015947404e+01f,  2.029225272e+01f,  2.026691796e+01f,  1.790897120e+01f,  1.985965212e+01f, 
            1.998823272e+01f,  1.996428708e+01f,  2.873340984e+01f,  3.169695762e+01f,  3.191747508e+01f,  3.187247886e+01f, 
            2.832331152e+01f,  3.125030058e+01f,  3.146455548e+01f,  3.142162566e+01f,  5.531836848e+01f,  6.114323556e+01f, 
            6.154860456e+01f,  6.146973180e+01f,  5.449364448e+01f,  6.024376980e+01f,  6.063654456e+01f,  6.056183916e+01f, 
            7.972771176e+01f,  8.830192374e+01f,  8.885606364e+01f,  8.875464138e+01f,  7.848383472e+01f,  8.694349758e+01f, 
            8.747864244e+01f,  8.738352306e+01f,  8.058263544e+01f,  8.947898118e+01f,  9.003760524e+01f,  8.993296890e+01f, 
            7.932517632e+01f,  8.810209998e+01f,  8.864152164e+01f,  8.854329186e+01f,  8.143755912e+01f,  9.065603862e+01f, 
            9.121914684e+01f,  9.111129642e+01f,  8.016651792e+01f,  8.926070238e+01f,  8.980440084e+01f,  8.970306066e+01f, 
            5.199837936e+01f,  5.798982564e+01f,  5.832735336e+01f,  5.826767868e+01f,  5.114649120e+01f,  5.705344980e+01f, 
            5.737796856e+01f,  5.732266860e+01f,  2.484347160e+01f,  2.775884274e+01f,  2.790852948e+01f,  2.788487406e+01f, 
            2.441526384e+01f,  2.728757898e+01f,  2.743072668e+01f,  2.740927590e+01f,  3.528770080e+01f,  3.925073304e+01f, 
            3.948209904e+01f,  3.943957416e+01f,  3.471747136e+01f,  3.862648248e+01f,  3.884917584e+01f,  3.880956744e+01f, 
            6.751588160e+01f,  7.525449264e+01f,  7.566690528e+01f,  7.559806416e+01f,  6.636938624e+01f,  7.399778928e+01f, 
            7.439276448e+01f,  7.432980240e+01f,  9.664832352e+01f,  1.079620654e+02f,  1.085046523e+02f,  1.084259801e+02f, 
            9.491952576e+01f,  1.060647070e+02f,  1.065809995e+02f,  1.065112150e+02f,  9.762523680e+01f,  1.093100148e+02f, 
            1.098560923e+02f,  1.097743788e+02f,  9.587832960e+01f,  1.073880497e+02f,  1.079075563e+02f,  1.078348687e+02f, 
            9.860215008e+01f,  1.106579642e+02f,  1.112075323e+02f,  1.111227775e+02f,  9.683713344e+01f,  1.087113924e+02f, 
            1.092341131e+02f,  1.091585225e+02f,  6.248207936e+01f,  7.026253104e+01f,  7.057564128e+01f,  7.053682128e+01f, 
            6.129936512e+01f,  6.895661424e+01f,  6.925173408e+01f,  6.921906960e+01f,  2.960262304e+01f,  3.336013848e+01f, 
            3.348987504e+01f,  3.347939880e+01f,  2.900824768e+01f,  3.270307896e+01f,  3.282377424e+01f,  3.281639880e+01f, 
            2.502357088e+01f,  2.801422296e+01f,  2.808948144e+01f,  2.809945320e+01f,  2.445334144e+01f,  2.738997240e+01f, 
            2.745655824e+01f,  2.746944648e+01f,  4.687896512e+01f,  5.263383216e+01f,  5.273237088e+01f,  5.276935248e+01f, 
            4.573246976e+01f,  5.137712880e+01f,  5.145823008e+01f,  5.150109072e+01f,  6.552996384e+01f,  7.380961416e+01f, 
            7.387890192e+01f,  7.396020792e+01f,  6.380116608e+01f,  7.191225576e+01f,  7.195524912e+01f,  7.204544280e+01f, 
            6.618090720e+01f,  7.471464264e+01f,  7.478244432e+01f,  7.486319736e+01f,  6.443400000e+01f,  7.279267752e+01f, 
            7.283390832e+01f,  7.292368728e+01f,  6.683185056e+01f,  7.561967112e+01f,  7.568598672e+01f,  7.576618680e+01f, 
            6.506683392e+01f,  7.367309928e+01f,  7.371256752e+01f,  7.380193176e+01f,  4.119322304e+01f,  4.675602864e+01f, 
            4.674531168e+01f,  4.681729104e+01f,  4.001050880e+01f,  4.545011184e+01f,  4.542140448e+01f,  4.549953936e+01f, 
            1.890386656e+01f,  2.153306712e+01f,  2.150006064e+01f,  2.154539880e+01f,  1.830949120e+01f,  2.087600760e+01f, 
            2.083395984e+01f,  2.088239880e+01f,  1.512929064e+01f,  1.707235074e+01f,  1.706293188e+01f,  1.709294814e+01f, 
            1.468404480e+01f,  1.658263194e+01f,  1.656646668e+01f,  1.659879126e+01f,  2.773626000e+01f,  3.141419076e+01f, 
            3.135429576e+01f,  3.142814364e+01f,  2.684124096e+01f,  3.042860148e+01f,  3.035514456e+01f,  3.043364364e+01f, 
            3.779374392e+01f,  4.298860998e+01f,  4.283676684e+01f,  4.296846906e+01f,  3.644442432e+01f,  4.150099854e+01f, 
            4.132870884e+01f,  4.146743970e+01f,  3.815971272e+01f,  4.350128598e+01f,  4.334646204e+01f,  4.347868266e+01f, 
            3.679681104e+01f,  4.199521950e+01f,  4.181974164e+01f,  4.195909458e+01f,  3.852568152e+01f,  4.401396198e+01f, 
            4.385615724e+01f,  4.398889626e+01f,  3.714919776e+01f,  4.248944046e+01f,  4.231077444e+01f,  4.245074946e+01f, 
            2.301659088e+01f,  2.641527684e+01f,  2.626680456e+01f,  2.637021852e+01f,  2.209440768e+01f,  2.539277748e+01f, 
            2.523032856e+01f,  2.533860108e+01f,  1.016564232e+01f,  1.173165282e+01f,  1.163564388e+01f,  1.169488062e+01f, 
            9.702287040e+00f,  1.121732730e+01f,  1.111429548e+01f,  1.117597878e+01f,  7.520012000e+00f,  8.583775800e+00f, 
            8.531652720e+00f,  8.567656200e+00f,  7.211465600e+00f,  8.242942680e+00f,  8.186160720e+00f,  8.223783720e+00f, 
            1.326101728e+01f,  1.522417752e+01f,  1.509117744e+01f,  1.517308584e+01f,  1.264090624e+01f,  1.453841016e+01f, 
            1.439604624e+01f,  1.448121672e+01f,  1.720490640e+01f,  1.989659844e+01f,  1.965369096e+01f,  1.979154396e+01f, 
            1.627021248e+01f,  1.886179572e+01f,  1.860477336e+01f,  1.874755404e+01f,  1.736739312e+01f,  2.012765220e+01f, 
            1.988151336e+01f,  2.002033404e+01f,  1.642364448e+01f,  1.908054612e+01f,  1.882015416e+01f,  1.896397164e+01f, 
            1.752987984e+01f,  2.035870596e+01f,  2.010933576e+01f,  2.024912412e+01f,  1.657707648e+01f,  1.929929652e+01f, 
            1.903553496e+01f,  1.918038924e+01f,  9.810996160e+00f,  1.149786072e+01f,  1.130138544e+01f,  1.140521640e+01f, 
            9.172775680e+00f,  1.078748664e+01f,  1.058137104e+01f,  1.068860232e+01f,  3.961666400e+00f,  4.703423160e+00f, 
            4.589979120e+00f,  4.647260040e+00f,  3.641047040e+00f,  4.346185560e+00f,  4.227898320e+00f,  4.286890920e+00f, 
            2.406620080e+00f,  2.806868700e+00f,  2.756917560e+00f,  2.783399460e+00f,  2.246488960e+00f,  2.629275180e+00f, 
            2.576913960e+00f,  2.604245940e+00f,  3.875007200e+00f,  4.580533560e+00f,  4.465563120e+00f,  4.523823240e+00f, 
            3.553235840e+00f,  4.223295960e+00f,  4.103482320e+00f,  4.163454120e+00f,  4.396106640e+00f,  5.308691220e+00f, 
            5.113495080e+00f,  5.208898860e+00f,  3.911185920e+00f,  4.769758980e+00f,  4.567263480e+00f,  4.665252060e+00f, 
            4.436603760e+00f,  5.368852980e+00f,  5.171419080e+00f,  5.267617740e+00f,  3.947155680e+00f,  4.823769060e+00f, 
            4.618966680e+00f,  4.717784700e+00f,  4.477100880e+00f,  5.429014740e+00f,  5.229343080e+00f,  5.326336620e+00f, 
            3.983125440e+00f,  4.877779140e+00f,  4.670669880e+00f,  4.770317340e+00f,  1.998209120e+00f,  2.520521400e+00f, 
            2.371601520e+00f,  2.441928840e+00f,  1.667383040e+00f,  2.150980440e+00f,  1.997079120e+00f,  2.069187240e+00f, 
            5.028239200e-01f,  7.067487000e-01f,  6.243399600e-01f,  6.623591400e-01f,  3.366563200e-01f,  5.209529400e-01f, 
            4.360419600e-01f,  4.749573000e-01f,  2.003826472e+01f,  2.189567814e+01f,  2.205388476e+01f,  2.200130490e+01f, 
            1.980803440e+01f,  2.164631502e+01f,  2.180130516e+01f,  2.174997858e+01f,  3.882635792e+01f,  4.247517516e+01f, 
            4.277334072e+01f,  4.267506708e+01f,  3.836438816e+01f,  4.197439836e+01f,  4.226610792e+01f,  4.217035236e+01f, 
            5.635522488e+01f,  6.172618770e+01f,  6.214592628e+01f,  6.200891406e+01f,  5.566000656e+01f,  6.097194666e+01f, 
            6.138196668e+01f,  6.124874886e+01f,  5.672169192e+01f,  6.222927042e+01f,  6.265174788e+01f,  6.251304222e+01f, 
            5.602194624e+01f,  6.146887770e+01f,  6.188156748e+01f,  6.174669078e+01f,  5.708815896e+01f,  6.273235314e+01f, 
            6.315756948e+01f,  6.301717038e+01f,  5.638388592e+01f,  6.196580874e+01f,  6.238116828e+01f,  6.224463270e+01f, 
            3.676030928e+01f,  4.043976972e+01f,  4.070426232e+01f,  4.061791764e+01f,  3.628928480e+01f,  3.992668956e+01f, 
            4.018458792e+01f,  4.010083044e+01f,  1.772790472e+01f,  1.952488422e+01f,  1.964759196e+01f,  1.960807002e+01f, 
            1.749163792e+01f,  1.926731886e+01f,  1.938671796e+01f,  1.934849538e+01f,  3.695151248e+01f,  4.056294732e+01f, 
            4.082867832e+01f,  4.074193044e+01f,  3.647933600e+01f,  4.004986716e+01f,  4.030900392e+01f,  4.022484324e+01f, 
            7.130522272e+01f,  7.837049880e+01f,  7.886408304e+01f,  7.870505064e+01f,  7.035785152e+01f,  7.734023736e+01f, 
            7.782058704e+01f,  7.766675208e+01f,  1.030430213e+02f,  1.133980477e+02f,  1.140813310e+02f,  1.138646156e+02f, 
            1.016174371e+02f,  1.118465039e+02f,  1.125098662e+02f,  1.123009816e+02f,  1.036944629e+02f,  1.142934829e+02f, 
            1.149809998e+02f,  1.147615196e+02f,  1.022598240e+02f,  1.127296357e+02f,  1.133970934e+02f,  1.131855131e+02f, 
            1.043459045e+02f,  1.151889181e+02f,  1.158806686e+02f,  1.156584236e+02f,  1.029022109e+02f,  1.136127676e+02f, 
            1.142843206e+02f,  1.140700446e+02f,  6.686955040e+01f,  7.390598040e+01f,  7.432779504e+01f,  7.419483240e+01f, 
            6.590406976e+01f,  7.285111224e+01f,  7.325941584e+01f,  7.313178888e+01f,  3.208154576e+01f,  3.550147212e+01f, 
            3.569261112e+01f,  3.563377620e+01f,  3.159729632e+01f,  3.497198748e+01f,  3.515634792e+01f,  3.510019236e+01f, 
            5.052885816e+01f,  5.574343698e+01f,  5.606310708e+01f,  5.596205454e+01f,  4.980301968e+01f,  5.495228586e+01f, 
            5.526182268e+01f,  5.516477190e+01f,  9.701482416e+01f,  1.071692298e+02f,  1.077496798e+02f,  1.075703065e+02f, 
            9.555861984e+01f,  1.055807759e+02f,  1.061408902e+02f,  1.059695550e+02f,  1.394307338e+02f,  1.542404684e+02f, 
            1.550223932e+02f,  1.547876385e+02f,  1.372396363e+02f,  1.518485600e+02f,  1.525998776e+02f,  1.523772319e+02f, 
            1.402856575e+02f,  1.554175258e+02f,  1.562039348e+02f,  1.559659660e+02f,  1.380809779e+02f,  1.530071624e+02f, 
            1.537627568e+02f,  1.535370007e+02f,  1.411405812e+02f,  1.565945833e+02f,  1.573854764e+02f,  1.571442935e+02f, 
            1.389223195e+02f,  1.541657648e+02f,  1.549256360e+02f,  1.546967695e+02f,  8.990595312e+01f,  9.988189092e+01f, 
            1.003480510e+02f,  1.002111001e+02f,  8.842258464e+01f,  9.825652692e+01f,  9.870193656e+01f,  9.857323116e+01f, 
            4.285003800e+01f,  4.767139314e+01f,  4.787378388e+01f,  4.781729646e+01f,  4.210609008e+01f,  4.685563530e+01f, 
            4.704761628e+01f,  4.699526886e+01f,  6.055941664e+01f,  6.717877656e+01f,  6.749589744e+01f,  6.740185512e+01f, 
            5.956820032e+01f,  6.609520056e+01f,  6.639848784e+01f,  6.630994248e+01f,  1.155333920e+02f,  1.283546270e+02f, 
            1.289075837e+02f,  1.287511906e+02f,  1.135449229e+02f,  1.261792728e+02f,  1.267044701e+02f,  1.265591170e+02f, 
            1.648857072e+02f,  1.834783380e+02f,  1.841852923e+02f,  1.839985164e+02f,  1.618939488e+02f,  1.802030033e+02f, 
            1.808681803e+02f,  1.806980335e+02f,  1.658626205e+02f,  1.848262874e+02f,  1.855367323e+02f,  1.853469151e+02f, 
            1.628527526e+02f,  1.815263460e+02f,  1.821947371e+02f,  1.820216873e+02f,  1.668395338e+02f,  1.861742369e+02f, 
            1.868881723e+02f,  1.866953138e+02f,  1.638115565e+02f,  1.828496887e+02f,  1.835212939e+02f,  1.833453410e+02f, 
            1.054477472e+02f,  1.178507602e+02f,  1.182424829e+02f,  1.181470766e+02f,  1.034230592e+02f,  1.156261925e+02f, 
            1.159896029e+02f,  1.159055131e+02f,  4.982249632e+01f,  5.577627672e+01f,  5.592983664e+01f,  5.589880872e+01f, 
            4.880713408e+01f,  5.465989176e+01f,  5.479924944e+01f,  5.477390280e+01f,  4.271752288e+01f,  4.767440856e+01f, 
            4.774252464e+01f,  4.774742760e+01f,  4.172630656e+01f,  4.659083256e+01f,  4.664511504e+01f,  4.665551496e+01f, 
            7.974094784e+01f,  8.919825072e+01f,  8.925153888e+01f,  8.929386576e+01f,  7.775247872e+01f,  8.702289648e+01f, 
            8.704842528e+01f,  8.710179216e+01f,  1.110340560e+02f,  1.245223130e+02f,  1.244772763e+02f,  1.245898246e+02f, 
            1.080422976e+02f,  1.212469783e+02f,  1.211601643e+02f,  1.212893417e+02f,  1.116849994e+02f,  1.254273415e+02f, 
            1.253808187e+02f,  1.254928140e+02f,  1.086751315e+02f,  1.221274001e+02f,  1.220388235e+02f,  1.221675862e+02f, 
            1.123359427e+02f,  1.263323700e+02f,  1.262843611e+02f,  1.263958034e+02f,  1.093079654e+02f,  1.230078218e+02f, 
            1.229174827e+02f,  1.230458306e+02f,  6.900336320e+01f,  7.780854192e+01f,  7.769064288e+01f,  7.779893328e+01f, 
            6.697867520e+01f,  7.558397424e+01f,  7.543776288e+01f,  7.555736976e+01f,  3.154597600e+01f,  3.568134744e+01f, 
            3.557926704e+01f,  3.565050216e+01f,  3.053061376e+01f,  3.456496248e+01f,  3.444867984e+01f,  3.452559624e+01f, 
            2.555809320e+01f,  2.871704322e+01f,  2.866743108e+01f,  2.871106398e+01f,  2.479710720e+01f,  2.788283034e+01f, 
            2.782260108e+01f,  2.787047766e+01f,  4.669942416e+01f,  5.263661124e+01f,  5.247310536e+01f,  5.258579868e+01f, 
            4.517292480e+01f,  5.096203380e+01f,  5.077722456e+01f,  5.089843980e+01f,  6.339682872e+01f,  7.172179398e+01f, 
            7.137969804e+01f,  7.158708666e+01f,  6.110028864e+01f,  6.920070030e+01f,  6.882654564e+01f,  6.904676898e+01f, 
            6.376279752e+01f,  7.223446998e+01f,  7.188939324e+01f,  7.209730026e+01f,  6.145267536e+01f,  6.969492126e+01f, 
            6.931757844e+01f,  6.953842386e+01f,  6.412876632e+01f,  7.274714598e+01f,  7.239908844e+01f,  7.260751386e+01f, 
            6.180506208e+01f,  7.018914222e+01f,  6.980861124e+01f,  7.003007874e+01f,  3.819087312e+01f,  4.350376836e+01f, 
            4.320523656e+01f,  4.337072028e+01f,  3.663720960e+01f,  4.179228084e+01f,  4.147203096e+01f,  4.164624396e+01f, 
            1.680556296e+01f,  1.924241634e+01f,  1.905976548e+01f,  1.915584318e+01f,  1.602646752e+01f,  1.838359674e+01f, 
            1.819005228e+01f,  1.829051190e+01f,  1.257810608e+01f,  1.427993964e+01f,  1.417779672e+01f,  1.423449012e+01f, 
            1.205906624e+01f,  1.370944380e+01f,  1.360006152e+01f,  1.365966468e+01f,  2.211424480e+01f,  2.523852888e+01f, 
            2.499000624e+01f,  2.512103592e+01f,  2.107314688e+01f,  2.409343608e+01f,  2.383038864e+01f,  2.396726088e+01f, 
            2.859030672e+01f,  3.285116100e+01f,  3.241174536e+01f,  3.263489244e+01f,  2.702413248e+01f,  3.112737012e+01f, 
            3.066609816e+01f,  3.089804364e+01f,  2.875279344e+01f,  3.308221476e+01f,  3.263956776e+01f,  3.286368252e+01f, 
            2.717756448e+01f,  3.134612052e+01f,  3.088147896e+01f,  3.111446124e+01f,  2.891528016e+01f,  3.331326852e+01f, 
            3.286739016e+01f,  3.309247260e+01f,  2.733099648e+01f,  3.156487092e+01f,  3.109685976e+01f,  3.133087884e+01f, 
            1.613830240e+01f,  1.875625944e+01f,  1.841329584e+01f,  1.858173096e+01f,  1.507909504e+01f,  1.758655992e+01f, 
            1.722879504e+01f,  1.740321096e+01f,  6.493839200e+00f,  7.643634360e+00f,  7.449204720e+00f,  7.542658440e+00f, 
            5.962726400e+00f,  7.056734040e+00f,  6.854880720e+00f,  6.951336360e+00f,  3.988446640e+00f,  4.621468380e+00f, 
            4.534895160e+00f,  4.577528100e+00f,  3.723068800e+00f,  4.329043500e+00f,  4.238769960e+00f,  4.282898100e+00f, 
            6.407180000e+00f,  7.520744760e+00f,  7.324788720e+00f,  7.419221640e+00f,  5.874915200e+00f,  6.933844440e+00f, 
            6.730464720e+00f,  6.827899560e+00f,  7.247145360e+00f,  8.685525780e+00f,  8.357239080e+00f,  8.512708140e+00f, 
            6.446484480e+00f,  7.802099460e+00f,  7.462642680e+00f,  7.622631900e+00f,  7.287642480e+00f,  8.745687540e+00f, 
            8.415163080e+00f,  8.571427020e+00f,  6.482454240e+00f,  7.856109540e+00f,  7.514345880e+00f,  7.675164540e+00f, 
            7.328139600e+00f,  8.805849300e+00f,  8.473087080e+00f,  8.630145900e+00f,  6.518424000e+00f,  7.910119620e+00f, 
            7.566049080e+00f,  7.727697180e+00f,  3.267421280e+00f,  4.082756280e+00f,  3.837367920e+00f,  3.951609480e+00f, 
            2.726101760e+00f,  3.483552600e+00f,  3.230602320e+00f,  3.347914920e+00f,  8.216898400e-01f,  1.143372060e+00f, 
            1.008858360e+00f,  1.070770020e+00f,  5.502755200e-01f,  8.427449400e-01f,  7.044387600e-01f,  7.678917000e-01f, 
        };

        float[] gw_expect = new float[] {
            5.922993204e+02f,  6.535818024e+02f,  6.497776620e+02f,  6.463643808e+02f,  5.953690404e+02f,  6.569841912e+02f, 
            6.531652380e+02f,  6.497222544e+02f,  5.625672084e+02f,  6.243912648e+02f,  6.205951116e+02f,  6.171672960e+02f, 
            5.654787204e+02f,  6.276388248e+02f,  6.238276284e+02f,  6.203705712e+02f,  5.328350964e+02f,  5.952007272e+02f, 
            5.914125612e+02f,  5.879702112e+02f,  5.355884004e+02f,  5.982934584e+02f,  5.944900188e+02f,  5.910188880e+02f, 
            5.984387604e+02f,  6.603865800e+02f,  6.565528140e+02f,  6.530801280e+02f,  6.015084804e+02f,  6.637889688e+02f, 
            6.599403900e+02f,  6.564380016e+02f,  5.683902324e+02f,  6.308863848e+02f,  6.270601452e+02f,  6.235738464e+02f, 
            5.713017444e+02f,  6.341339448e+02f,  6.302926620e+02f,  6.267771216e+02f,  5.383417044e+02f,  6.013861896e+02f, 
            5.975674764e+02f,  5.940675648e+02f,  5.410950084e+02f,  6.044789208e+02f,  6.006449340e+02f,  5.971162416e+02f, 
            6.045782004e+02f,  6.671913576e+02f,  6.633279660e+02f,  6.597958752e+02f,  6.076479204e+02f,  6.705937464e+02f, 
            6.667155420e+02f,  6.631537488e+02f,  5.742132564e+02f,  6.373815048e+02f,  6.335251788e+02f,  6.299803968e+02f, 
            5.771247684e+02f,  6.406290648e+02f,  6.367576956e+02f,  6.331836720e+02f,  5.438483124e+02f,  6.075716520e+02f, 
            6.037223916e+02f,  6.001649184e+02f,  5.466016164e+02f,  6.106643832e+02f,  6.067998492e+02f,  6.032135952e+02f, 
            6.352754004e+02f,  7.012152456e+02f,  6.972037260e+02f,  6.933746112e+02f,  6.383451204e+02f,  7.046176344e+02f, 
            7.005913020e+02f,  6.967324848e+02f,  6.033283764e+02f,  6.698571048e+02f,  6.658503468e+02f,  6.620131488e+02f, 
            6.062398884e+02f,  6.731046648e+02f,  6.690828636e+02f,  6.652164240e+02f,  5.713813524e+02f,  6.384989640e+02f, 
            6.344969676e+02f,  6.306516864e+02f,  5.741346564e+02f,  6.415916952e+02f,  6.375744252e+02f,  6.337003632e+02f, 
            6.414148404e+02f,  7.080200232e+02f,  7.039788780e+02f,  7.000903584e+02f,  6.444845604e+02f,  7.114224120e+02f, 
            7.073664540e+02f,  7.034482320e+02f,  6.091514004e+02f,  6.763522248e+02f,  6.723153804e+02f,  6.684196992e+02f, 
            6.120629124e+02f,  6.795997848e+02f,  6.755478972e+02f,  6.716229744e+02f,  5.768879604e+02f,  6.446844264e+02f, 
            6.406518828e+02f,  6.367490400e+02f,  5.796412644e+02f,  6.477771576e+02f,  6.437293404e+02f,  6.397977168e+02f, 
            6.475542804e+02f,  7.148248008e+02f,  7.107540300e+02f,  7.068061056e+02f,  6.506240004e+02f,  7.182271896e+02f, 
            7.141416060e+02f,  7.101639792e+02f,  6.149744244e+02f,  6.828473448e+02f,  6.787804140e+02f,  6.748262496e+02f, 
            6.178859364e+02f,  6.860949048e+02f,  6.820129308e+02f,  6.780295248e+02f,  5.823945684e+02f,  6.508698888e+02f, 
            6.468067980e+02f,  6.428463936e+02f,  5.851478724e+02f,  6.539626200e+02f,  6.498842556e+02f,  6.458950704e+02f, 
            6.782514804e+02f,  7.488486888e+02f,  7.446297900e+02f,  7.403848416e+02f,  6.813212004e+02f,  7.522510776e+02f, 
            7.480173660e+02f,  7.437427152e+02f,  6.440895444e+02f,  7.153229448e+02f,  7.111055820e+02f,  7.068590016e+02f, 
            6.470010564e+02f,  7.185705048e+02f,  7.143380988e+02f,  7.100622768e+02f,  6.099276084e+02f,  6.817972008e+02f, 
            6.775813740e+02f,  6.733331616e+02f,  6.126809124e+02f,  6.848899320e+02f,  6.806588316e+02f,  6.763818384e+02f, 
            6.843909204e+02f,  7.556534664e+02f,  7.514049420e+02f,  7.471005888e+02f,  6.874606404e+02f,  7.590558552e+02f, 
            7.547925180e+02f,  7.504584624e+02f,  6.499125684e+02f,  7.218180648e+02f,  7.175706156e+02f,  7.132655520e+02f, 
            6.528240804e+02f,  7.250656248e+02f,  7.208031324e+02f,  7.164688272e+02f,  6.154342164e+02f,  6.879826632e+02f, 
            6.837362892e+02f,  6.794305152e+02f,  6.181875204e+02f,  6.910753944e+02f,  6.868137468e+02f,  6.824791920e+02f, 
            6.905303604e+02f,  7.624582440e+02f,  7.581800940e+02f,  7.538163360e+02f,  6.936000804e+02f,  7.658606328e+02f, 
            7.615676700e+02f,  7.571742096e+02f,  6.557355924e+02f,  7.283131848e+02f,  7.240356492e+02f,  7.196721024e+02f, 
            6.586471044e+02f,  7.315607448e+02f,  7.272681660e+02f,  7.228753776e+02f,  6.209408244e+02f,  6.941681256e+02f, 
            6.898912044e+02f,  6.855278688e+02f,  6.236941284e+02f,  6.972608568e+02f,  6.929686620e+02f,  6.885765456e+02f, 
            7.212275604e+02f,  7.964821320e+02f,  7.920558540e+02f,  7.873950720e+02f,  7.242972804e+02f,  7.998845208e+02f, 
            7.954434300e+02f,  7.907529456e+02f,  6.848507124e+02f,  7.607887848e+02f,  7.563608172e+02f,  7.517048544e+02f, 
            6.877622244e+02f,  7.640363448e+02f,  7.595933340e+02f,  7.549081296e+02f,  6.484738644e+02f,  7.250954376e+02f, 
            7.206657804e+02f,  7.160146368e+02f,  6.512271684e+02f,  7.281881688e+02f,  7.237432380e+02f,  7.190633136e+02f, 
            7.273670004e+02f,  8.032869096e+02f,  7.988310060e+02f,  7.941108192e+02f,  7.304367204e+02f,  8.066892984e+02f, 
            8.022185820e+02f,  7.974686928e+02f,  6.906737364e+02f,  7.672839048e+02f,  7.628258508e+02f,  7.581114048e+02f, 
            6.935852484e+02f,  7.705314648e+02f,  7.660583676e+02f,  7.613146800e+02f,  6.539804724e+02f,  7.312809000e+02f, 
            7.268206956e+02f,  7.221119904e+02f,  6.567337764e+02f,  7.343736312e+02f,  7.298981532e+02f,  7.251606672e+02f, 
            7.335064404e+02f,  8.100916872e+02f,  8.056061580e+02f,  8.008265664e+02f,  7.365761604e+02f,  8.134940760e+02f, 
            8.089937340e+02f,  8.041844400e+02f,  6.964967604e+02f,  7.737790248e+02f,  7.692908844e+02f,  7.645179552e+02f, 
            6.994082724e+02f,  7.770265848e+02f,  7.725234012e+02f,  7.677212304e+02f,  6.594870804e+02f,  7.374663624e+02f, 
            7.329756108e+02f,  7.282093440e+02f,  6.622403844e+02f,  7.405590936e+02f,  7.360530684e+02f,  7.312580208e+02f, 
            7.642036404e+02f,  8.441155752e+02f,  8.394819180e+02f,  8.344053024e+02f,  7.672733604e+02f,  8.475179640e+02f, 
            8.428694940e+02f,  8.377631760e+02f,  7.256118804e+02f,  8.062546248e+02f,  8.016160524e+02f,  7.965507072e+02f, 
            7.285233924e+02f,  8.095021848e+02f,  8.048485692e+02f,  7.997539824e+02f,  6.870201204e+02f,  7.683936744e+02f, 
            7.637501868e+02f,  7.586961120e+02f,  6.897734244e+02f,  7.714864056e+02f,  7.668276444e+02f,  7.617447888e+02f, 
            7.703430804e+02f,  8.509203528e+02f,  8.462570700e+02f,  8.411210496e+02f,  7.734128004e+02f,  8.543227416e+02f, 
            8.496446460e+02f,  8.444789232e+02f,  7.314349044e+02f,  8.127497448e+02f,  8.080810860e+02f,  8.029572576e+02f, 
            7.343464164e+02f,  8.159973048e+02f,  8.113136028e+02f,  8.061605328e+02f,  6.925267284e+02f,  7.745791368e+02f, 
            7.699051020e+02f,  7.647934656e+02f,  6.952800324e+02f,  7.776718680e+02f,  7.729825596e+02f,  7.678421424e+02f, 
            7.764825204e+02f,  8.577251304e+02f,  8.530322220e+02f,  8.478367968e+02f,  7.795522404e+02f,  8.611275192e+02f, 
            8.564197980e+02f,  8.511946704e+02f,  7.372579284e+02f,  8.192448648e+02f,  8.145461196e+02f,  8.093638080e+02f, 
            7.401694404e+02f,  8.224924248e+02f,  8.177786364e+02f,  8.125670832e+02f,  6.980333364e+02f,  7.807645992e+02f, 
            7.760600172e+02f,  7.708908192e+02f,  7.007866404e+02f,  7.838573304e+02f,  7.791374748e+02f,  7.739394960e+02f, 
        };
    }
}
