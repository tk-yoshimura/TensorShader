using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>QuaternionCast</summary>
        public static Field QuaternionCast(Field real, Field imag_i, Field imag_j, Field imag_k) {
            Field y = new();
            Link link = new Links.Quaternion.QuaternionCast(real, imag_i, imag_j, imag_k, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Quaternion {
    /// <summary>QuaternionCast</summary>
    public class QuaternionCast : Link {
        /// <summary>実部項</summary>
        protected Field XReal => InFields[0];

        /// <summary>第1虚部項</summary>
        protected Field XImagI => InFields[1];

        /// <summary>第2虚部項</summary>
        protected Field XImagJ => InFields[2];

        /// <summary>第3虚部項</summary>
        protected Field XImagK => InFields[3];

        /// <summary>複素数項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public QuaternionCast(Field realfield, Field imagifield, Field imagjfield, Field imagkfield, Field outfield)
            : base(new Field[] { realfield, imagifield, imagjfield, imagkfield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(QuaternionCast(XReal.Value, XImagI.Value, XImagJ.Value, XImagK.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (XReal.EnableBackprop) {
                XReal.AddGrad(QuaternionR(Y.Grad));
            }

            if (XImagI.EnableBackprop) {
                XImagI.AddGrad(QuaternionI(Y.Grad));
            }

            if (XImagJ.EnableBackprop) {
                XImagJ.AddGrad(QuaternionJ(Y.Grad));
            }

            if (XImagK.EnableBackprop) {
                XImagK.AddGrad(QuaternionK(Y.Grad));
            }
        }
    }
}
