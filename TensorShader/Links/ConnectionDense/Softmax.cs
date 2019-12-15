using System;
using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>Softmax</summary>
        public static Field Softmax(Field x) {
            Field y = new Field();
            Link link = new Links.ConnectionDense.Softmax(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ConnectionDense {
    /// <summary>Softmax</summary>
    public class Softmax : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public Softmax(Field infield, Field outfield)
            : base(new Field[] { infield }, outfield) {
            if (infield.Shape.Type != ShapeType.Map || infield.Shape.Ndim != 2) {
                throw new ArgumentException(ExceptionMessage.TensorElements(infield.Shape, ("Ndim", 2), ("Type", ShapeType.Map)));
            }
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            VariableNode x_exp = Exp(X.Value);
            VariableNode x_exp_sum = Sum(x_exp, new int[] { Axis.Map0D.Channels }, keepdims: true);
            Y.AssignValue(x_exp / Broadcast(x_exp_sum, X.Shape));
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
