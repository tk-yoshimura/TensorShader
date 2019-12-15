using System;
using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>2乗冪</summary>
        public static Field Exp2(Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Exp2(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>2乗冪</summary>
    internal class Exp2 : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Exp2(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Exp2(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad((float)Math.Log(2.0) * Y.Value * Y.Grad);
            }
        }
    }
}
