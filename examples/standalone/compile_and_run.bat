bazel build --config=windows_cpp17 //:beginner_cc

bazel-bin\beginner_cc.exe --dataset_dir=..\..\yggdrasil_decision_forests\test_data\dataset --output_dir=result
