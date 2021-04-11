using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>対数関数</summary>
        public static Field Log(Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Log(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>対数関数</summary>
    internal class Log : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Log(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Log(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Rcp(X.Value) * Y.Grad);
            }
        }
    }
}
