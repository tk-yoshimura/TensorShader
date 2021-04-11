using System;
using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>Hardmax</summary>
        public static Field Hardmax(Field x) {
            Field y = new();
            Link link = new Links.Channelwise.Hardmax(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Channelwise {
    /// <summary>Hardmax</summary>
    public class Hardmax : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public Hardmax(Field infield, Field outfield)
            : base(new Field[] { infield }, outfield) {
            if (infield.Shape.Type != ShapeType.Map) {
                throw new ArgumentException(ExceptionMessage.ShapeType(infield.Shape.Type, ShapeType.Map));
            }
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Hardmax(X.Value));
        }
    }
}
