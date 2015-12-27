bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package --genrule_strategy=standalone --spawn_strategy=standalone --verbose_failures || exit 1
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

source /data-local/wchan/tensorflow_env/bin/activate
pip install --upgrade /tmp/tensorflow_pkg/*
