using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>Clamp</summary>
        public static Field Clamp(Field x, float xmin, float xmax) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Clamp(x, y, xmin, xmax);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>Clamp</summary>
    internal class Clamp : UnaryArithmetric {
        /// <summary>定数</summary>
        public float Min { private set; get; }

        /// <summary>定数</summary>
        public float Max { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Clamp(Field infield, Field outfield, float xmin, float xmax)
            : base(infield, outfield) {

            this.Min = xmin;
            this.Max = xmax;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Clamp(X.Value, Min, Max));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Y.Grad * LessThanOrEqual(Min, X.Value) * LessThanOrEqual(X.Value, Max));
            }
        }
    }
}
