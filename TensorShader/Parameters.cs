using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace TensorShader {
    /// <summary>パラメータコンテナ</summary>
    public class Parameters {
        private readonly List<ParameterField> parameter_fields;

        /// <summary>パラメータ数</summary>
        public int Count => parameter_fields.Count;

        /// <summary>コンストラクタ</summary>
        public Parameters(IEnumerable<ParameterField> parameter_fields) {
            this.parameter_fields = new List<ParameterField>(parameter_fields);
        }

        /// <summary>リストからキャスト</summary>
        public static implicit operator Parameters(List<ParameterField> parameter_fields) {
            return new Parameters(parameter_fields);
        }

        /// <summary>リストへキャスト</summary>
        public static implicit operator List<ParameterField>(Parameters parameters) {
            return new List<ParameterField>(parameters.parameter_fields);
        }

        /// <summary>条件抜き出し</summary>
        public Parameters Where(Func<ParameterField, bool> predicate) {
            return new Parameters(parameter_fields.Where(predicate));
        }

        /// <summary>更新則を追加</summary>
        public void AddUpdater(Func<ParameterField, Updater> gen_updater) {
            if (parameter_fields.Count <= 0) {
                throw new InvalidOperationException(ExceptionMessage.EmptyList());
            }

            foreach (ParameterField parameter_field in parameter_fields) {
                parameter_field.AddUpdater(gen_updater(parameter_field));
            }
        }

        /// <summary>更新則を初期化</summary>
        public void InitializeUpdater() {
            if (parameter_fields.Count <= 0) {
                throw new InvalidOperationException(ExceptionMessage.EmptyList());
            }

            foreach (ParameterField parameter_field in parameter_fields) {
                foreach (Updater updater in parameter_field.Updaters) {
                    updater.Initialize();
                }
            }
        }

        /// <summary>更新</summary>
        public void Update() {
            if (parameter_fields.Count <= 0) {
                throw new InvalidOperationException(ExceptionMessage.EmptyList());
            }

            foreach (ParameterField parameter_field in parameter_fields) {
                parameter_field.Update();
            }
        }

        /// <summary>初期化</summary>
        public void InitializeTensor(Func<Tensor, Initializer> initializer) {
            if (parameter_fields.Count <= 0) {
                throw new InvalidOperationException(ExceptionMessage.EmptyList());
            }

            foreach (ParameterField parameter_field in parameter_fields) {
                parameter_field.Initialize(initializer);
            }
        }

        /// <summary>更新則のパラメータ変更・取得</summary>
        /// <param name="name">値識別名(クラス名.プロパティ名)</param>
        public object this[string name] {
            set {
                string[] name_split = name.Split('.');

                if (name_split.Length != 2) {
                     throw new FormatException(ExceptionMessage.InvalidParamKey());
                }

                string class_name = name.Split('.')[0];
                string property_name = name.Split('.')[1];

                bool has_changed = false;

                foreach (ParameterField parameter_field in parameter_fields) {
                    foreach (Updater updater in parameter_field.Updaters) {
                        if (updater.Name != class_name) {
                            continue;
                        }

                        PropertyInfo prop_info = updater.GetType().GetProperty(property_name);
                        if (prop_info == null) {
                            continue;
                        }

                        if (prop_info.PropertyType == value.GetType()) {
                            prop_info.SetValue(updater, value);
                            has_changed = true;
                        }
                        else if (prop_info.PropertyType == typeof(float) && value.GetType() == typeof(double)) {
                            prop_info.SetValue(updater, (float)(double)value);
                            has_changed = true;
                        }
                    }
                }

                if (!has_changed) {
                    throw new KeyNotFoundException(name);
                }
            }

            get {
                string[] name_split = name.Split('.');

                if (name_split.Length != 2) {
                    throw new FormatException(ExceptionMessage.InvalidParamKey());
                }

                List<object> values = new List<object>();

                string class_name = name.Split('.')[0];
                string property_name = name.Split('.')[1];

                foreach (ParameterField parameter_field in parameter_fields) {
                    foreach (Updater updater in parameter_field.Updaters) {
                        if (updater.Name != class_name) {
                            continue;
                        }

                        PropertyInfo prop_info = updater.GetType().GetProperty(property_name);
                        if (prop_info == null) {
                            continue;
                        }

                        object value = prop_info.GetValue(updater);
                        if (value == null) {
                            continue;
                        }

                        values.Add(value);
                    }
                }

                if (values.Count < 1) {
                    throw new KeyNotFoundException(name);
                }

                if (values.Distinct().Count() != 1) {
                    throw new ArgumentException(ExceptionMessage.ContainsSeveralDifferentValues(name));
                }

                return values[0];
            }
        }

        /// <summary>スナップショットに状態保存</summary>
        public Snapshot Save() {
            Snapshot snapshot = new Snapshot();

            List<string> used_keys = new List<string>();

            foreach (ParameterField parameter_field in parameter_fields) {
                string key = parameter_field.Name != string.Empty ? parameter_field.Name : "unnamed";
                int duplicated_index = 0;

                while (used_keys.Contains(key)) {
                    duplicated_index++;
                    key = $"{parameter_field.Name}_{duplicated_index}";
                }

                used_keys.Add(key);

                if (parameter_field.ValueTensor != null) {
                    string key_value = key + "/value";
                    snapshot.Append(key_value, parameter_field.ValueTensor);
                }

                if (parameter_field.GradTensor != null) {
                    string key_grad = key + "/grad";
                    snapshot.Append(key_grad, parameter_field.GradTensor);
                }

                foreach (Updater updater in parameter_field.Updaters) {
                    foreach (var item in updater.States) {
                        if (item.Value != null) {
                            string key_updaterstate = key + "/updater_" + updater.Name + '/' + item.Key;
                            snapshot.Append(key_updaterstate, item.Value);
                        }

                    }
                }
            }

            return snapshot;
        }

        /// <summary>スナップショットから状態展開</summary>
        public void Load(Snapshot snapshot) {
            var table = snapshot.Table;

            List<string> used_keys = new List<string>();

            foreach (ParameterField parameter_field in parameter_fields) {
                string key = parameter_field.Name != string.Empty ? parameter_field.Name : "unnamed";
                int duplicated_index = 0;

                while (used_keys.Contains(key)) {
                    duplicated_index++;
                    key = $"{parameter_field.Name}_{duplicated_index}";
                }

                used_keys.Add(key);

                string key_value = key + "/value";
                if (parameter_field.ValueTensor != null && table.ContainsKey(key_value)) {
                    snapshot.Load(key_value, parameter_field.ValueTensor);
                }

                string key_grad = key + "/grad";
                if (parameter_field.GradTensor != null && table.ContainsKey(key_grad)) {
                    snapshot.Load(key_grad, parameter_field.GradTensor);
                }

                foreach (Updater updater in parameter_field.Updaters) {
                    foreach (var item in updater.States) {
                        string key_updaterstate = key + "/updater_" + updater.Name + '/' + item.Key;
                        if (item.Value != null && table.ContainsKey(key_updaterstate)) {
                            snapshot.Load(key_updaterstate, item.Value);
                        }

                    }
                }
            }
        }

        /// <summary>スナップショットにValueのみ保存</summary>
        public Snapshot Freeze() {
            Snapshot snapshot = new Snapshot();

            List<string> used_keys = new List<string>();

            foreach (ParameterField parameter_field in parameter_fields) {
                string key = parameter_field.Name != string.Empty ? parameter_field.Name : "unnamed";
                int duplicated_index = 0;

                while (used_keys.Contains(key)) {
                    duplicated_index++;
                    key = $"{parameter_field.Name}_{duplicated_index}";
                }

                used_keys.Add(key);

                if (parameter_field.ValueTensor != null) {
                    string key_value = key + "/value";
                    snapshot.Append(key_value, parameter_field.ValueTensor);
                }
            }

            return snapshot;
        }
    }
}
