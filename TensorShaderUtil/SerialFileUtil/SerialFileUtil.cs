using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace TensorShaderUtil {

    /// <summary>連番ファイルユーティリティ</summary>
    public static class SerialFileUtil {

        /// <summary>連番ファイルを列挙</summary>
        /// <param name="directory_path">ディレクトリパス</param>
        /// <param name="filename_pattern">ファイル名パターン e.g. ( test*.txt )</param>
        /// <returns>ファイルパス及び連番</returns>
        public static IEnumerable<(string path, long num)> EnumrateSerialFiles(string directory_path, string filename_pattern) {
            if (!filename_pattern.Contains('*') || filename_pattern.IndexOf('*', filename_pattern.IndexOf('*') + 1) >= 0) {
                throw new ArgumentException(null, nameof(filename_pattern));
            }

            var regex = new Regex($"^{filename_pattern.Replace("*", "(?<num>\\d+)")}$");

            var filepaths = Directory.EnumerateFiles(directory_path, filename_pattern);
            var filepaths_serials = filepaths.Select((filepath) => {
                string filename = Path.GetFileName(filepath);
                Match match = regex.Match(filename);

                if (match.Success && long.TryParse(match.Groups["num"].Value, out long num)) {
                    return (filepath, num);
                }
                return (filepath, num: long.MinValue);
            }).Where((item) => item.num != long.MinValue);

            return filepaths_serials;
        }

        /// <summary>最大の連番ファイルを返す</summary>
        /// <param name="directory_path">ディレクトリパス</param>
        /// <param name="filename_pattern">ファイル名パターン e.g. ( test*.txt )</param>
        /// <remarks>対象ファイルが見つからないとき(path:null, num:0)を返す</remarks>
        public static (string path, long num) SearchLastSerialFile(string directory_path, string filename_pattern) {
            var filepaths = EnumrateSerialFiles(directory_path, filename_pattern);
            return filepaths.OrderBy((filepath) => filepath.num).LastOrDefault();
        }

        /// <summary>最小の連番ファイルを返す</summary>
        /// <param name="directory_path">ディレクトリパス</param>
        /// <param name="filename_pattern">ファイル名パターン e.g. ( test*.txt )</param>
        /// <remarks>対象ファイルが見つからないとき(path:null, num:0)を返す</remarks>
        public static (string path, long num) SearchFirstSerialFile(string directory_path, string filename_pattern) {
            var filepaths = EnumrateSerialFiles(directory_path, filename_pattern);
            return filepaths.OrderBy((filepath) => filepath.num).FirstOrDefault();
        }
    }
}
