using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>範囲内に収める</summary>
        public static Field Clip(Field x, float cmin, float cmax) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Clip(x, y, cmin, cmax);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>範囲内に収める</summary>
    internal class Clip : UnaryArithmetric {
        /// <summary>最大値</summary>
        public float ConstantMax { private set; get; }

        /// <summary>最小値</summary>
        public float ConstantMin { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Clip(Field infield, Field outfield, float cmin, float cmax)
            : base(infield, outfield) {
            this.ConstantMin = cmin;
            this.ConstantMax = cmax;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Clip(X.Value, ConstantMin, ConstantMax));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Y.Grad * Equal(X.Value, Y.Value));
            }
        }
    }
}
