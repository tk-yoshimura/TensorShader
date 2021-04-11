using System;
using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>2進対数</summary>
        public static Field Log2(Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Log2(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>2進対数</summary>
    internal class Log2 : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Log2(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Log2(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Rcp((float)Math.Log(2.0) * X.Value) * Y.Grad);
            }
        }
    }
}
