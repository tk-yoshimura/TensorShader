using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>ComplexCast</summary>
        public static Field ComplexCast(Field real, Field imag) {
            Field y = new Field();
            Link link = new Links.Complex.ComplexCast(real, imag, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Complex {
    /// <summary>ComplexCast</summary>
    public class ComplexCast : Link {
        /// <summary>実部項</summary>
        protected Field XReal => InFields[0];

        /// <summary>虚部項</summary>
        protected Field XImag => InFields[1];

        /// <summary>複素数項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public ComplexCast(Field realfield, Field imagfield, Field outfield)
            : base(new Field[] { realfield, imagfield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ComplexCast(XReal.Value, XImag.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (XReal.EnableBackprop) {
                XReal.AddGrad(ComplexReal(Y.Grad));
            }

            if (XImag.EnableBackprop) {
                XImag.AddGrad(ComplexImag(Y.Grad));
            }
        }
    }
}
