using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>TrivectorCast</summary>
        public static Field TrivectorCast(Field x, Field y, Field z) {
            Field v = new();
            Link link = new Links.Trivector.TrivectorCast(x, y, z, v);

            link.Forward();

            return v;
        }
    }
}

namespace TensorShader.Links.Trivector {
    /// <summary>TrivectorCast</summary>
    public class TrivectorCast : Link {
        /// <summary>X成分項</summary>
        protected Field X => InFields[0];

        /// <summary>Y成分項</summary>
        protected Field Y => InFields[1];

        /// <summary>Z成分項</summary>
        protected Field Z => InFields[2];

        /// <summary>出力項</summary>
        protected Field V => OutField;

        /// <summary>コンストラクタ</summary>
        public TrivectorCast(Field xfield, Field yfield, Field zfield, Field outfield)
            : base(new Field[] { xfield, yfield, zfield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            V.AssignValue(TrivectorCast(X.Value, Y.Value, Z.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (V.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(TrivectorX(V.Grad));
            }

            if (Y.EnableBackprop) {
                Y.AddGrad(TrivectorY(V.Grad));
            }

            if (Z.EnableBackprop) {
                Z.AddGrad(TrivectorZ(V.Grad));
            }
        }
    }
}
