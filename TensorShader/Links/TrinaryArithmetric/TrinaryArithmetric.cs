namespace TensorShader.Links.TrinaryArithmetric {
    /// <summary>3項演算</summary>
    public abstract class TrinaryArithmetric : Link {
        /// <summary>入力項1</summary>
        protected Field X1 => InFields[0];

        /// <summary>入力項2</summary>
        protected Field X2 => InFields[1];

        /// <summary>入力項3</summary>
        protected Field X3 => InFields[2];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public TrinaryArithmetric(Field infield1, Field infield2, Field infield3, Field outfield)
            : base(new Field[] { infield1, infield2, infield3 }, outfield) { }
    }
}
