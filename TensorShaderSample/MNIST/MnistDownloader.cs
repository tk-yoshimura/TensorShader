using System;
using System.IO;
using System.Net.Http;

namespace MNIST {
    public static class MnistDownloader {
        public static async void Download(string dirpath_dataset) {
            if (!Directory.Exists(dirpath_dataset)) {
                Directory.CreateDirectory(dirpath_dataset);
            }

            string url = "http://yann.lecun.com/exdb/mnist/";

            Console.WriteLine($"Download from : {url}");

            string[] filenames = {
                "train-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz"
            };

            using (var client = new HttpClient()) {
                foreach (string filename in filenames) {
                    string filepath = $"{dirpath_dataset}/{filename}";
                    string filepath_temp = $"{dirpath_dataset}/{filename}.download";

                    if (File.Exists(filepath_temp)) {
                        File.Delete(filepath_temp);
                    }

                    if (!File.Exists(filepath)) {
                        using HttpResponseMessage res = client.GetAsync($"{url}{filename}").Result;
                        using (Stream stream = res.Content.ReadAsStreamAsync().Result) {
                            using FileStream fs = new(filepath_temp, FileMode.CreateNew);
                            stream.CopyTo(fs);
                        }

                        File.Move(filepath_temp, filepath);
                    }
                }
            }
        }
    }
}
