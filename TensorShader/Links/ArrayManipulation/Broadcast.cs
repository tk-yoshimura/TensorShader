using System;
using System.Collections.Generic;
using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>ブロードキャスト</summary>
        public static Field Broadcast(Field x, Shape shape) {
            Field y = new();
            Link link = new Links.ArrayManipulation.Broadcast(x, y, shape);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ArrayManipulation {
    /// <summary>ブロードキャスト</summary>
    public class Broadcast : Link {
        private readonly int[] axes_keepdims, axes_abandondims;

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>入力形状</summary>
        protected Shape InShape { private set; get; }

        /// <summary>出力形状</summary>
        protected Shape OutShape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Broadcast(Field infield, Field outfield, Shape shape)
            : base(new Field[] { infield }, outfield) {
            if (shape.Ndim < infield.Shape.Ndim) {
                throw new ArgumentException(ExceptionMessage.Broadcast(infield.Shape, shape));
            }

            List<int> axes_keepdims = new();
            List<int> axes_abandondims = new();

            for (int i = 0; i < infield.Shape.Ndim; i++) {
                if (infield.Shape[i] == 1 && shape[i] > 1) {
                    axes_keepdims.Add(i);
                }
            }
            for (int i = infield.Shape.Ndim; i < shape.Ndim; i++) {
                axes_abandondims.Add(i);
            }

            this.axes_keepdims = axes_keepdims.ToArray();
            this.axes_abandondims = axes_abandondims.ToArray();
            this.InShape = infield.Shape;
            this.OutShape = shape;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Broadcast(X.Value, OutShape));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                VariableNode gsum = Sum(Sum(Y.Grad, axes_keepdims, keepdims: true), axes_abandondims, keepdims: false);

                X.AddGrad(Broadcast(gsum, InShape));
            }
        }
    }
}
