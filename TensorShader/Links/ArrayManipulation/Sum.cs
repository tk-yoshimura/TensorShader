using System.Collections.Generic;
using System.Linq;
using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>テンソル総和</summary>
        public static Field Sum(params Field[] xs) {
            if (xs.Length == 1) {
                return xs[0];
            }
            else if (xs.Length == 2) {
                return xs[0] + xs[1];
            }

            Field y = new Field();
            Link link = new Links.ArrayManipulation.Sum(xs, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ArrayManipulation {
    /// <summary>テンソル総和</summary>
    public class Sum : Link {
        /// <summary>入力項</summary>
        protected IReadOnlyList<Field> XS => InFields;

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public Sum(Field[] infields, Field outfield)
            : base(infields, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Sum(XS.Select((field) => field.Value).ToArray()));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            for (int i = 0; i < XS.Count; i++) {
                if (XS[i].EnableBackprop) {
                    XS[i].AddGrad(Y.Grad);
                }
            }
        }
    }
}
