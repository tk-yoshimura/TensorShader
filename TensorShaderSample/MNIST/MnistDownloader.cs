using System;
using System.IO;
using System.Net;

namespace MNIST {
    public static class MnistDownloader {
        public static void Download(string dirpath_dataset) {
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

            using (var wc = new WebClient()) {
                foreach (string filename in filenames) {
                    string filepath = $"{dirpath_dataset}/{filename}";
                    string filepath_temp = $"{dirpath_dataset}/{filename}.download";

                    if (File.Exists(filepath_temp)){
                        File.Delete(filepath_temp);
                    }

                    if (!File.Exists(filepath)) {
                        wc.DownloadFile($"{url}{filename}", filepath_temp);
                        File.Move(filepath_temp, filepath);
                    }
                }
            }
        }
    }
}
