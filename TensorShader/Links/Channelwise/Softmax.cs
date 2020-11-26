using System;
using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>Softmax</summary>
        public static Field Softmax(Field x) {
            Field y = new Field();
            Link link = new Links.Channelwise.Softmax(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Channelwise {
    /// <summary>Softmax</summary>
    public class Softmax : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public Softmax(Field infield, Field outfield)
            : base(new Field[] { infield }, outfield) {
            if (infield.Shape.Type != ShapeType.Map) {
                throw new ArgumentException(ExceptionMessage.TensorType(infield.Shape.Type, ShapeType.Map));
            }
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Softmax(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                VariableNode gx = Y.Value * Y.Grad;
                VariableNode gxsum = Sum(gx, new int[] { Axis.Map0D.Channels }, keepdims: true);

                X.AddGrad(gx - Y.Value * Broadcast(gxsum, X.Shape));
            }
        }
    }
}
