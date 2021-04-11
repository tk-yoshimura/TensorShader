using System;
using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>常用対数</summary>
        public static Field Log10(Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Log10(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>常用対数</summary>
    internal class Log10 : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Log10(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Log10(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Rcp((float)Math.Log(10.0) * X.Value) * Y.Grad);
            }
        }
    }
}
