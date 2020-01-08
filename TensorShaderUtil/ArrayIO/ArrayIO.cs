using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace TensorShaderUtil.ArrayIO {

    /// <summary>配列ファイル読み込み</summary>
    public static class ArrayIO {

        /// <summary>読み込み</summary>
        /// <param name="filepath">ファイルパス</param>
        /// <param name="skip_rows">スキップ行数</param>
        public static string[] Read1D(string filepath, int skip_rows = 0) {
            if (skip_rows < 0) {
                throw new ArgumentOutOfRangeException(nameof(skip_rows));
            }

            List<string> vals = new List<string>();

            using (var stream = new StreamReader(filepath)) {
                for (int i = 0; i < skip_rows; i++) {
                    stream.ReadLine();
                }

                while (!stream.EndOfStream) {
                    string line = stream.ReadLine();

                    if (string.IsNullOrEmpty(line)) {
                        break;
                    }

                    vals.Add(line);
                }
            }

            return vals.ToArray();
        }

        /// <summary>読み込み</summary>
        /// <param name="filepath">ファイルパス</param>
        /// <param name="skip_rows">スキップ行数</param>
        /// <param name="skip_cols">スキップ列数</param>
        /// <param name="delimiter">区切り文字</param>
        public static (string[] vals, int rows, int cols) Read2D(string filepath, int skip_rows = 0, int skip_cols = 0, char delimiter = ',') {
            if (skip_rows < 0) {
                throw new ArgumentOutOfRangeException(nameof(skip_rows));
            }
            if (skip_cols < 0) {
                throw new ArgumentOutOfRangeException(nameof(skip_cols));
            }

            List<string> vals = new List<string>();
            int rows = 0, cols = 0;

            using (var stream = new StreamReader(filepath)) {
                for (int i = 0; i < skip_rows; i++) {
                    stream.ReadLine();
                }

                while (!stream.EndOfStream) {
                    string line = stream.ReadLine();

                    if (string.IsNullOrEmpty(line)) {
                        break;
                    }

                    var line_splits = line.Split(new char[]{ delimiter }, StringSplitOptions.RemoveEmptyEntries)
                                          .Skip(skip_cols);

                    if (cols > 0 && cols != line_splits.Count()) {
                        throw new InvalidDataException();
                    }

                    rows++;
                    cols = line_splits.Count();

                    vals.AddRange(line_splits);
                }
            }

            return (vals.ToArray(), rows, cols);
        }

        /// <summary>書き込み</summary>
        /// <param name="filepath">ファイルパス</param>
        /// <param name="strs">文字配列</param>
        public static void Write1D(string filepath, string[] strs) {
            using (var stream = new StreamWriter(filepath)) {
                foreach (string str in strs) {
                    stream.WriteLine(str);
                }
            }
        }

        /// <summary>書き込み</summary>
        /// <param name="filepath">ファイルパス</param>
        /// <param name="strs">文字配列</param>
        /// <param name="rows">行数</param>
        /// <param name="cols">列数</param>
        /// <param name="delimiter">区切り文字</param>
        public static void Write2D(string filepath, string[] strs, int rows, int cols, char delimiter = ',') {
            if(rows < 0 || cols < 0 || strs.Length != rows * cols) { 
                throw new ArgumentException($"{nameof(rows)},{nameof(cols)}");
            }

            using (var stream = new StreamWriter(filepath)) {
                for (int i = 0, k = 0; k < strs.Length; k += cols) {
                    for (int j = 0; j < cols; i++, j++) {
                        string str = strs[i];

                        stream.Write($"{str}{delimiter}");
                    }

                    stream.Write(Environment.NewLine);
                }
            }
        }
    }
}
