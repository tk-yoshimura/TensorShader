using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>全象限対応逆正接関数</summary>
        public static Field Arctan2(Field y, Field x) {
            Field theta = new Field();
            Link link = new Links.BinaryArithmetric.Arctan2(y, x, theta);

            link.Forward();

            return theta;
        }
    }
}

namespace TensorShader.Links.BinaryArithmetric {
    /// <summary>全象限対応逆正接関数</summary>
    internal class Arctan2 : BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Arctan2(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Arctan2(X1.Value, X2.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X1.EnableBackprop || X2.EnableBackprop) {
                VariableNode rnorm = Rcp(Square(X1.Value) + Square(X2.Value) + 1e-5f);

                if (X1.EnableBackprop) {
                    X1.AddGrad(Y.Grad * X2.Value * rnorm);
                }

                if (X2.EnableBackprop) {
                    X2.AddGrad(-Y.Grad * X1.Value * rnorm);
                }
            }
        }
    }
}
