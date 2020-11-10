using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection1D {
    [TestClass]
    public class ConvolutionTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, kwidth = 3, inwidth = 13;
            int outwidth = inwidth - kwidth + 1, batch = 2;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            ParameterField x = (Shape.Map1D(inchannels, inwidth, batch), xval);
            ParameterField w = (Shape.Kernel1D(inchannels, outchannels, kwidth), wval);
            VariableField y_actual = (Shape.Map1D(outchannels, outwidth, batch), yval);

            Field y_expect = Convolution1D(x, w);
            Field err = Abs(y_expect - y_actual);

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;
            float[] gw_actual = w.GradState;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = new float[] {
            2.661230000e-02f, 2.648558000e-02f, 2.635886000e-02f, 2.623214000e-02f, 2.610542000e-02f, 2.597870000e-02f, 2.585198000e-02f,
            5.692571500e-02f, 5.660732000e-02f, 5.628892500e-02f, 5.597053000e-02f, 5.565213500e-02f, 5.533374000e-02f, 5.501534500e-02f,
            8.593871000e-02f, 8.536368500e-02f, 8.478866000e-02f, 8.421363500e-02f, 8.363861000e-02f, 8.306358500e-02f, 8.248856000e-02f,
            1.113097700e-01f, 1.105398800e-01f, 1.097699900e-01f, 1.090001000e-01f, 1.082302100e-01f, 1.074603200e-01f, 1.066904300e-01f,
            1.366808300e-01f, 1.357160750e-01f, 1.347513200e-01f, 1.337865650e-01f, 1.328218100e-01f, 1.318570550e-01f, 1.308923000e-01f,
            1.620518900e-01f, 1.608922700e-01f, 1.597326500e-01f, 1.585730300e-01f, 1.574134100e-01f, 1.562537900e-01f, 1.550941700e-01f,
            1.874229500e-01f, 1.860684650e-01f, 1.847139800e-01f, 1.833594950e-01f, 1.820050100e-01f, 1.806505250e-01f, 1.792960400e-01f,
            2.127940100e-01f, 2.112446600e-01f, 2.096953100e-01f, 2.081459600e-01f, 2.065966100e-01f, 2.050472600e-01f, 2.034979100e-01f,
            2.381650700e-01f, 2.364208550e-01f, 2.346766400e-01f, 2.329324250e-01f, 2.311882100e-01f, 2.294439950e-01f, 2.276997800e-01f,
            2.635361300e-01f, 2.615970500e-01f, 2.596579700e-01f, 2.577188900e-01f, 2.557798100e-01f, 2.538407300e-01f, 2.519016500e-01f,
            2.889071900e-01f, 2.867732450e-01f, 2.846393000e-01f, 2.825053550e-01f, 2.803714100e-01f, 2.782374650e-01f, 2.761035200e-01f,
            1.396218450e-01f, 1.381342600e-01f, 1.366466750e-01f, 1.351590900e-01f, 1.336715050e-01f, 1.321839200e-01f, 1.306963350e-01f,
            4.165227000e-02f, 4.087600000e-02f, 4.009973000e-02f, 3.932346000e-02f, 3.854719000e-02f, 3.777092000e-02f, 3.699465000e-02f,
            2.487635150e-01f, 2.475503800e-01f, 2.463372450e-01f, 2.451241100e-01f, 2.439109750e-01f, 2.426978400e-01f, 2.414847050e-01f,
            4.175741900e-01f, 4.150829650e-01f, 4.125917400e-01f, 4.101005150e-01f, 4.076092900e-01f, 4.051180650e-01f, 4.026268400e-01f,
            5.014304900e-01f, 4.975962200e-01f, 4.937619500e-01f, 4.899276800e-01f, 4.860934100e-01f, 4.822591400e-01f, 4.784248700e-01f,
            5.268015500e-01f, 5.227724150e-01f, 5.187432800e-01f, 5.147141450e-01f, 5.106850100e-01f, 5.066558750e-01f, 5.026267400e-01f,
            5.521726100e-01f, 5.479486100e-01f, 5.437246100e-01f, 5.395006100e-01f, 5.352766100e-01f, 5.310526100e-01f, 5.268286100e-01f,
            5.775436700e-01f, 5.731248050e-01f, 5.687059400e-01f, 5.642870750e-01f, 5.598682100e-01f, 5.554493450e-01f, 5.510304800e-01f,
            6.029147300e-01f, 5.983010000e-01f, 5.936872700e-01f, 5.890735400e-01f, 5.844598100e-01f, 5.798460800e-01f, 5.752323500e-01f,
            6.282857900e-01f, 6.234771950e-01f, 6.186686000e-01f, 6.138600050e-01f, 6.090514100e-01f, 6.042428150e-01f, 5.994342200e-01f,
            6.536568500e-01f, 6.486533900e-01f, 6.436499300e-01f, 6.386464700e-01f, 6.336430100e-01f, 6.286395500e-01f, 6.236360900e-01f,
            6.790279100e-01f, 6.738295850e-01f, 6.686312600e-01f, 6.634329350e-01f, 6.582346100e-01f, 6.530362850e-01f, 6.478379600e-01f,
            7.043989700e-01f, 6.990057800e-01f, 6.936125900e-01f, 6.882194000e-01f, 6.828262100e-01f, 6.774330200e-01f, 6.720398300e-01f,
            3.329624100e-01f, 3.293019950e-01f, 3.256415800e-01f, 3.219811650e-01f, 3.183207500e-01f, 3.146603350e-01f, 3.109999200e-01f,
            9.649557500e-02f, 9.463289000e-02f, 9.277020500e-02f, 9.090752000e-02f, 8.904483500e-02f, 8.718215000e-02f, 8.531946500e-02f,
        };

        float[] gw_expect = new float[] {
            3.867294200e-01f, 3.902922100e-01f, 3.938550000e-01f, 3.974177900e-01f, 4.009805800e-01f, 4.045433700e-01f, 4.081061600e-01f,
            3.531180730e-01f, 3.563661860e-01f, 3.596142990e-01f, 3.628624120e-01f, 3.661105250e-01f, 3.693586380e-01f, 3.726067510e-01f,
            3.195067260e-01f, 3.224401620e-01f, 3.253735980e-01f, 3.283070340e-01f, 3.312404700e-01f, 3.341739060e-01f, 3.371073420e-01f,
            2.858953790e-01f, 2.885141380e-01f, 2.911328970e-01f, 2.937516560e-01f, 2.963704150e-01f, 2.989891740e-01f, 3.016079330e-01f,
            2.522840320e-01f, 2.545881140e-01f, 2.568921960e-01f, 2.591962780e-01f, 2.615003600e-01f, 2.638044420e-01f, 2.661085240e-01f,
            2.186726850e-01f, 2.206620900e-01f, 2.226514950e-01f, 2.246409000e-01f, 2.266303050e-01f, 2.286197100e-01f, 2.306091150e-01f,
            1.850613380e-01f, 1.867360660e-01f, 1.884107940e-01f, 1.900855220e-01f, 1.917602500e-01f, 1.934349780e-01f, 1.951097060e-01f,
            1.514499910e-01f, 1.528100420e-01f, 1.541700930e-01f, 1.555301440e-01f, 1.568901950e-01f, 1.582502460e-01f, 1.596102970e-01f,
            1.178386440e-01f, 1.188840180e-01f, 1.199293920e-01f, 1.209747660e-01f, 1.220201400e-01f, 1.230655140e-01f, 1.241108880e-01f,
            8.422729700e-02f, 8.495799400e-02f, 8.568869100e-02f, 8.641938800e-02f, 8.715008500e-02f, 8.788078200e-02f, 8.861147900e-02f,
            5.061595000e-02f, 5.103197000e-02f, 5.144799000e-02f, 5.186401000e-02f, 5.228003000e-02f, 5.269605000e-02f, 5.311207000e-02f,
            4.116689500e-01f, 4.152317400e-01f, 4.187945300e-01f, 4.223573200e-01f, 4.259201100e-01f, 4.294829000e-01f, 4.330456900e-01f,
            3.758548640e-01f, 3.791029770e-01f, 3.823510900e-01f, 3.855992030e-01f, 3.888473160e-01f, 3.920954290e-01f, 3.953435420e-01f,
            3.400407780e-01f, 3.429742140e-01f, 3.459076500e-01f, 3.488410860e-01f, 3.517745220e-01f, 3.547079580e-01f, 3.576413940e-01f,
            3.042266920e-01f, 3.068454510e-01f, 3.094642100e-01f, 3.120829690e-01f, 3.147017280e-01f, 3.173204870e-01f, 3.199392460e-01f,
            2.684126060e-01f, 2.707166880e-01f, 2.730207700e-01f, 2.753248520e-01f, 2.776289340e-01f, 2.799330160e-01f, 2.822370980e-01f,
            2.325985200e-01f, 2.345879250e-01f, 2.365773300e-01f, 2.385667350e-01f, 2.405561400e-01f, 2.425455450e-01f, 2.445349500e-01f,
            1.967844340e-01f, 1.984591620e-01f, 2.001338900e-01f, 2.018086180e-01f, 2.034833460e-01f, 2.051580740e-01f, 2.068328020e-01f,
            1.609703480e-01f, 1.623303990e-01f, 1.636904500e-01f, 1.650505010e-01f, 1.664105520e-01f, 1.677706030e-01f, 1.691306540e-01f,
            1.251562620e-01f, 1.262016360e-01f, 1.272470100e-01f, 1.282923840e-01f, 1.293377580e-01f, 1.303831320e-01f, 1.314285060e-01f,
            8.934217600e-02f, 9.007287300e-02f, 9.080357000e-02f, 9.153426700e-02f, 9.226496400e-02f, 9.299566100e-02f, 9.372635800e-02f,
            5.352809000e-02f, 5.394411000e-02f, 5.436013000e-02f, 5.477615000e-02f, 5.519217000e-02f, 5.560819000e-02f, 5.602421000e-02f,
            4.366084800e-01f, 4.401712700e-01f, 4.437340600e-01f, 4.472968500e-01f, 4.508596400e-01f, 4.544224300e-01f, 4.579852200e-01f,
            3.985916550e-01f, 4.018397680e-01f, 4.050878810e-01f, 4.083359940e-01f, 4.115841070e-01f, 4.148322200e-01f, 4.180803330e-01f,
            3.605748300e-01f, 3.635082660e-01f, 3.664417020e-01f, 3.693751380e-01f, 3.723085740e-01f, 3.752420100e-01f, 3.781754460e-01f,
            3.225580050e-01f, 3.251767640e-01f, 3.277955230e-01f, 3.304142820e-01f, 3.330330410e-01f, 3.356518000e-01f, 3.382705590e-01f,
            2.845411800e-01f, 2.868452620e-01f, 2.891493440e-01f, 2.914534260e-01f, 2.937575080e-01f, 2.960615900e-01f, 2.983656720e-01f,
            2.465243550e-01f, 2.485137600e-01f, 2.505031650e-01f, 2.524925700e-01f, 2.544819750e-01f, 2.564713800e-01f, 2.584607850e-01f,
            2.085075300e-01f, 2.101822580e-01f, 2.118569860e-01f, 2.135317140e-01f, 2.152064420e-01f, 2.168811700e-01f, 2.185558980e-01f,
            1.704907050e-01f, 1.718507560e-01f, 1.732108070e-01f, 1.745708580e-01f, 1.759309090e-01f, 1.772909600e-01f, 1.786510110e-01f,
            1.324738800e-01f, 1.335192540e-01f, 1.345646280e-01f, 1.356100020e-01f, 1.366553760e-01f, 1.377007500e-01f, 1.387461240e-01f,
            9.445705500e-02f, 9.518775200e-02f, 9.591844900e-02f, 9.664914600e-02f, 9.737984300e-02f, 9.811054000e-02f, 9.884123700e-02f,
            5.644023000e-02f, 5.685625000e-02f, 5.727227000e-02f, 5.768829000e-02f, 5.810431000e-02f, 5.852033000e-02f, 5.893635000e-02f,
        };
    }
}
