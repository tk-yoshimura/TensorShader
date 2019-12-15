using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>対数双曲線余弦関数</summary>
        public static Field LogCosh(Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.LogCosh(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>対数双曲線余弦関数</summary>
    internal class LogCosh : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public LogCosh(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(LogCosh(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Tanh(X.Value) * Y.Grad);
            }
        }
    }
}
