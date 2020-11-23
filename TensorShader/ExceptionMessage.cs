using System.Globalization;
using System.Linq;

namespace TensorShader {
    /// <summary>例外メッセージの生成</summary>
    public static class ExceptionMessage {
        private enum Lang { Default, JP }

        private static readonly Lang lang;

        static ExceptionMessage() {
            string culture_name = CultureInfo.CurrentCulture.Name;
            switch (culture_name) {
                case "ja-JP":
                    lang = Lang.JP;
                    break;
                default:
                    lang = Lang.Default;
                    break;
            }
        }

        /// <summary>テンソル形状の不正</summary>
        public static string Shape(Shape actual, Shape expected) {
            switch (lang) {
                case Lang.JP:
                    return $"テンソル形状が不正です。{expected}が想定されていますが、{actual}が与えられました。";
                default:
                    return $"Invalid shape. expected:{expected} actual:{actual}";
            }
        }

        /// <summary>テンソル形状の不正</summary>
        public static string Shape(string elementname, Shape shape) {
            switch (lang) {
                case Lang.JP:
                    return $"テンソル形状{shape}の{elementname}が不正です。";
                default:
                    return $"Invalid {elementname} of shape {shape}.";
            }
        }

        /// <summary>テンソル形状の不正</summary>
        public static string Shape(int[] shape) {
            switch (lang) {
                case Lang.JP:
                    return $"テンソル形状({string.Join(",", shape)})が不正です。";
                default:
                    return $"Invalid shape ({string.Join(",", shape)}).";
            }
        }

        /// <summary>テンソル長さ倍数</summary>
        public static string TensorLengthMultiple(string elementname, Shape shape, int actual, int multiple) {
            switch (lang) {
                case Lang.JP:
                    return $"テンソル形状{shape}の{elementname}が不正です。{multiple}の倍数が想定されていますが、{actual}が与えられました。";
                default:
                    return $"Invalid {elementname} of shape {shape}. expected:number of {multiple} actual:{actual}";
            }
        }

        /// <summary>テンソル形状の不正</summary>
        public static string ShapeWithIndex(int index, Shape actual, Shape expected) {
            switch (lang) {
                case Lang.JP:
                    return $"{index}番目のテンソル形状が不正です。{expected}が想定されていますが、{actual}が与えられました。";
                default:
                    return $"Invalid shape of {index}-th argument. expected:{expected} actual:{actual}";
            }
        }

        /// <summary>テンソル形状の不正</summary>
        public static string ShapeWithIndex(int index, string actual, string expected) {
            switch (lang) {
                case Lang.JP:
                    return $"{index}番目のテンソル形状が不正です。{expected}が想定されていますが、{actual}が与えられました。";
                default:
                    return $"Invalid shape of {index}-th argument. expected:{expected} actual:{actual}";
            }
        }

        /// <summary>軸長さの不正</summary>
        public static string AxisLength(string axisname, int actual, int expected) {
            switch (lang) {
                case Lang.JP:
                    return $"{axisname}軸が不正です。{expected}が想定されていますが、{actual}が与えられました。";
                default:
                    return $"Invalid {axisname} axis. expected:{expected} actual:{actual}";
            }
        }

        /// <summary>引数</summary>
        public static string Argument(string argname, int actual, int expected) {
            switch (lang) {
                case Lang.JP:
                    return $"{argname}が不正です。{expected}が想定されていますが、{actual}が与えられました。";
                default:
                    return $"Invalid {argname}. expected:{expected} actual:{actual}";
            }
        }

        /// <summary>引数の数</summary>
        public static string ArgumentCount(string argname, int actual, int expected) {
            switch (lang) {
                case Lang.JP:
                    return $"{argname}の数が不正です。{expected}が想定されていますが、{actual}が与えられました。";
                default:
                    return $"Invalid {argname} counts. expected:{expected} actual:{actual}";
            }
        }

        /// <summary>引数の約数</summary>
        public static string ArgumentMultiple(string elementname, int actual, int multiple) {
            switch (lang) {
                case Lang.JP:
                    return $"引数{elementname}が不正です。{multiple}の倍数が想定されていますが、{actual}が与えられました。";
                default:
                    return $"Invalid {elementname}. expected:number of {multiple} actual:{actual}";
            }
        }

        /// <summary>テンソル長さ</summary>
        public static string TensorLength(Shape actual, Shape expected) {
            switch (lang) {
                case Lang.JP:
                    return $"テンソルの長さが一致しません。{expected}, {actual}";
                default:
                    return $"Mismatch tensor length. {expected}, {actual}";
            }
        }

        /// <summary>テンソルタイプ</summary>
        public static string TensorType(ShapeType actual, ShapeType expected) {
            switch (lang) {
                case Lang.JP:
                    return $"テンソルのタイプが不正です。{expected}が想定されていますが、{actual}が与えられました。";
                default:
                    return $"Invalid tensor type. expected:{expected} actual:{actual}";
            }
        }

        /// <summary>テンソルの複数要素</summary>
        public static string TensorElements(Shape actual, params (string name, object obj)[] expected) {
            string str = string.Join(", ", expected.Select((item) => $"{item.name}={item.obj}"));

            switch (lang) {
                case Lang.JP:
                    return $"テンソルの形状が不正です。{str}であることが想定されていますが、{actual.Type} {actual}が与えられました。";
                default:
                    return $"Invalid tensor shape. expected:{str} actual:{actual.Type} {actual}";
            }
        }

        /// <summary>テンソルの複数要素</summary>
        public static string TensorElementsWithIndex(int index, Shape actual, params (string name, object obj)[] expected) {
            string str = string.Join(", ", expected.Select((item) => $"{item.name}={item.obj}"));

            switch (lang) {
                case Lang.JP:
                    return $"{index}番目のテンソルの形状が不正です。{str}であることが想定されていますが、{actual.Type} {actual}が与えられました。";
                default:
                    return $"Invalid shape of {index}-th argument. expected:{str} actual:{actual.Type} {actual}";
            }
        }

        /// <summary>ブロードキャスト不可</summary>
        public static string Broadcast(Shape specified, Shape broadcasted) {
            switch (lang) {
                case Lang.JP:
                    return $"与えられた形状{specified}は{broadcasted}にブロードキャストすることができません。";
                default:
                    return $"The shape specified for the node can't be broadcast. {specified}->{broadcasted}";
            }
        }

        /// <summary>ベクトル演算不可</summary>
        public static string Vectorize(Shape shape1, Shape shape2) {
            switch (lang) {
                case Lang.JP:
                    return $"与えられた形状{shape1}, {shape2}ではベクトル演算することができません。";
                default:
                    return $"The shape specified for the node can't be vectorize operation. {shape1}, {shape2}";
            }
        }

        /// <summary>結合不可</summary>
        public static string Concat(int axis, Shape[] inshapes, Shape outshape) {
            switch (lang) {
                case Lang.JP:
                    return $"与えられた形状{string.Join(", ", inshapes.Select((shape) => shape.ToString()))}は軸{axis}で{outshape}に結合することができません。";
                default:
                    return $"Invalid concat operation. {outshape} axis:{axis} <- {string.Join(", ", inshapes.Select((shape) => shape.ToString()))}";
            }
        }

        /// <summary>結合不可</summary>
        public static string Concat(int axis, Shape[] inshapes) {
            switch (lang) {
                case Lang.JP:
                    return $"与えられた形状{string.Join(", ", inshapes.Select((shape) => shape.ToString()))}は軸{axis}で結合することができません。";
                default:
                    return $"Invalid concat operation. {string.Join(", ", inshapes.Select((shape) => shape.ToString()))}";
            }
        }

        /// <summary>分離不可</summary>
        public static string Separate(int axis, Shape inshape, Shape[] outshapes) {
            switch (lang) {
                case Lang.JP:
                    return $"与えられた形状{inshape}は軸{axis}で{string.Join(", ", outshapes.Select((shape) => shape.ToString()))}に分離することができません。";
                default:
                    return $"Invalid separate operation. {inshape} axis:{axis} -> {string.Join(", ", outshapes.Select((shape) => shape.ToString()))}";
            }
        }

        /// <summary>空配列への操作</summary>
        public static string EmptyList() {
            switch (lang) {
                case Lang.JP:
                    return "空配列に操作を試みました。";
                default:
                    return "Attempted operation on an empty array.";
            }
        }

        /// <summary>不正なパラメータキー</summary>
        public static string InvalidParamKey() {
            switch (lang) {
                case Lang.JP:
                    return "パラメータ名は\"クラス名\".\"プロパティ名\"を指定してください。";
                default:
                    return "The name argument must be specified by \"class name\".\"property name\".";
            }
        }

        /// <summary>複数の異なるパラメータ値が含まれる</summary>
        public static string ContainsSeveralDifferentValues(string name) {
            switch (lang) {
                case Lang.JP:
                    return $"指定したパラメータキー {name} には複数の異なる値が含まれます。";
                default:
                    return $"Contains several different values : {name}.";
            }
        }
    }
}
