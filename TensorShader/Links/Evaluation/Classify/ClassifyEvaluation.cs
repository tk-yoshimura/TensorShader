using System;

namespace TensorShader.Links.Evaluation.Classify {
    /// <summary>クラス弁別評価量</summary>
    public abstract class ClassifyEvaluation : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>教師信号項</summary>
        protected Field T => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public ClassifyEvaluation(Field xfield, Field tfield, Field yfield)
            : base(new Field[] { xfield, tfield }, yfield) {
            if (X.Shape.Ndim != 2) {
                throw new ArgumentException(ExceptionMessage.ShapeElementsWithIndex(0, X.Shape, ("Ndim", 2)));
            }

            if (T.Shape.Ndim != 1) {
                throw new ArgumentException(ExceptionMessage.ShapeElementsWithIndex(1, T.Shape, ("Ndim", 1)));
            }
        }

        /// <summary>逆伝搬</summary>
        public override sealed void Backward() {
            return;
        }
    }
}
