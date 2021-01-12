using System.Collections.Generic;

namespace TensorShaderCudaBackend {

    /// <summary>配列操作</summary>
    public static class ArrayManipulation {
        private readonly static Dictionary<string, Shader> shaders = new Dictionary<string, Shader>();

        /// <summary>初期化</summary>
        public static void Clear(uint length, float c, CudaArray<float> dst, Stream stream = null) {
            string key = "clear";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Clear());
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, dst, c, length);
        }

        /// <summary>パターンコピー</summary>
        public static void PatternCopy(uint src_stride, uint src_index,
                                       uint dst_stride, uint dst_index,
                                       uint copy_length, uint slides,
                                       CudaArray<float> src, CudaArray<float> dst,
                                       Stream stream = null) {

            PatternCopy(src_offset: 0, src_stride, src_index, dst_offset: 0, dst_stride, dst_index, copy_length, slides, src, dst, stream);
        }

        /// <summary>パターンコピー</summary>
        public static void PatternCopy(uint src_offset, uint src_stride, uint src_index,
                                       uint dst_offset, uint dst_stride, uint dst_index,
                                       uint copy_length, uint slides,
                                       CudaArray<float> src, CudaArray<float> dst,
                                       Stream stream = null) {

            string key = "pattern_copy";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.PatternCopy());
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst, src_offset, src_index, dst_offset, dst_index, src_stride, dst_stride, copy_length, slides);

        }

        /// <summary>ブロードキャスト</summary>
        public static void Broadcast(uint src_stride, CudaArray<float> src, uint dst_stride, CudaArray<float> dst, uint slides, Stream stream = null) {
            string key = "broadcast";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Broadcast());
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst, src_stride, dst_stride, slides);
        }

        /// <summary>軸反転</summary>
        public static void Flip(uint stride, uint axislength, uint slides, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            string key = "flip";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.Flip());
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst, stride, axislength, slides);
        }

        /// <summary>ソート</summary>
        public static void Sort(uint stride, uint axislength, uint slides, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            Shader shader;

            if (stride > 1) {
                if (axislength <= Shaders.ArrayManipulation.SortUseSharedMemory.MaxAxisLength) {
                    string key = "sort_use_shared_memory";

                    if (!shaders.ContainsKey(key)) {
                        shaders.Add(key, new Shaders.ArrayManipulation.SortUseSharedMemory());
                    }

                    shader = shaders[key];
                }
                else {
                    string key = "sort_unuse_shared_memory";

                    if (!shaders.ContainsKey(key)) {
                        shaders.Add(key, new Shaders.ArrayManipulation.SortUnuseSharedMemory());
                    }

                    shader = shaders[key];
                }

                if (stream == null) {
                    stream = Shader.DefaultStream;
                }

                shader.Execute(stream, src, dst, stride, axislength, slides);
            }
            else {
                if (axislength <= Shaders.ArrayManipulation.SortUseSharedMemory.MaxAxisLength) {
                    string key = "sort_nostride_use_shared_memory";

                    if (!shaders.ContainsKey(key)) {
                        shaders.Add(key, new Shaders.ArrayManipulation.SortNoStrideUseSharedMemory());
                    }

                    shader = shaders[key];
                }
                else {
                    string key = "sort_nostride_unuse_shared_memory";

                    if (!shaders.ContainsKey(key)) {
                        shaders.Add(key, new Shaders.ArrayManipulation.SortNoStrideUnuseSharedMemory());
                    }

                    shader = shaders[key];
                }

                if (stream == null) {
                    stream = Shader.DefaultStream;
                }

                shader.Execute(stream, src, dst, axislength, slides);
            }
        }

        /// <summary>ソート(キーつき)</summary>
        public static void SortWithKey(uint stride, uint axislength, uint slides,
                                       CudaArray<float> src_key, CudaArray<float> src_value,
                                       CudaArray<float> dst_key, CudaArray<float> dst_value,
                                       Stream stream = null) {

            Shader shader;

            if (stride > 1) {
                if (axislength <= Shaders.ArrayManipulation.SortWithKeyUseSharedMemory.MaxAxisLength) {
                    string key = "sortwithkey_use_shared_memory";

                    if (!shaders.ContainsKey(key)) {
                        shaders.Add(key, new Shaders.ArrayManipulation.SortWithKeyUseSharedMemory());
                    }

                    shader = shaders[key];
                }
                else {
                    string key = "sortwithkey_unuse_shared_memory";

                    if (!shaders.ContainsKey(key)) {
                        shaders.Add(key, new Shaders.ArrayManipulation.SortWithKeyUnuseSharedMemory());
                    }

                    shader = shaders[key];
                }

                if (stream == null) {
                    stream = Shader.DefaultStream;
                }

                shader.Execute(stream, src_value, dst_value, src_key, dst_key, stride, axislength, slides);
            }
            else { 
                if (axislength <= Shaders.ArrayManipulation.SortWithKeyUseSharedMemory.MaxAxisLength) {
                    string key = "sortwithkey_nostride_use_shared_memory";

                    if (!shaders.ContainsKey(key)) {
                        shaders.Add(key, new Shaders.ArrayManipulation.SortWithKeyNoStrideUseSharedMemory());
                    }

                    shader = shaders[key];
                }
                else {
                    string key = "sortwithkey_nostride_unuse_shared_memory";

                    if (!shaders.ContainsKey(key)) {
                        shaders.Add(key, new Shaders.ArrayManipulation.SortWithKeyNoStrideUnuseSharedMemory());
                    }

                    shader = shaders[key];
                }

                if (stream == null) {
                    stream = Shader.DefaultStream;
                }

                shader.Execute(stream, src_value, dst_value, src_key, dst_key, axislength, slides);
            }
        }

        /// <summary>水平加算</summary>
        public static void HorizontalAdd(uint n, CudaArray<float> src, CudaArray<float> dst, Stream stream = null) {
            string key = "horizontal_add";

            if (!shaders.ContainsKey(key)) {
                shaders.Add(key, new Shaders.ArrayManipulation.HorizontalAdd());
            }

            Shader shader = shaders[key];

            if (stream == null) {
                stream = Shader.DefaultStream;
            }

            shader.Execute(stream, src, dst, n);
        }
    }
}
