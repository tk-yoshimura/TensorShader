using System;

namespace TensorShaderUtil {
    /// <summary>CRC32</summary>
    public static class Crc32 {
        private const int table_length = 256;
        private static readonly UInt32[] table;

        static Crc32() {
            table = new UInt32[table_length];

            unchecked {
                for (UInt32 i = 0; i < table_length; i++) {
                    UInt32 c = i;
                    for (int j = 0; j < 8; j++) {
                        c = ((c & 1) == 1) ? (0xEDB88320 ^ (c >> 1)) : (c >> 1);
                    }
                    table[i] = c;
                }
            }
        }

        /// <summary>ハッシュ値を計算</summary>
        public static UInt32 ComputeHash(byte[] data, long offset, long length) {
            UInt32 c = 0xFFFFFFFF;

            unchecked {
                for (long i = offset, i_end = offset + length; i < i_end; i++) {
                    c = table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
                }
            }

            return c ^ 0xFFFFFFFF;
        }
    }
}
