using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>形状変更</summary>
        public static Field Reshape(Field x, Shape shape) {
            Field y = new();
            Link link = new Links.ArrayManipulation.Reshape(x, y, shape);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ArrayManipulation {
    /// <summary>形状変更</summary>
    public class Reshape : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>入力形状</summary>
        protected Shape InShape { private set; get; }

        /// <summary>出力形状</summary>
        protected Shape OutShape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Reshape(Field infield, Field outfield, Shape shape)
            : base(new Field[] { infield }, outfield) {
            this.InShape = infield.Shape;
            this.OutShape = shape;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Reshape(X.Value, OutShape));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Reshape(Y.Grad, InShape));
            }
        }
    }
}
