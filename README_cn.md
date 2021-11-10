# Operator Capability Manager (OCM)

OCM 检查在英特尔平台（CPU、GPU、MYRIAD 和 HDDL）上使用 OpenVINO 运行深度学习模型的算子支持。目前它仅支持 Tensorflow 模型。


## 构建

要求：

- Tensorflow 版本：2.7.0
- Tensorflow CC 库
- Tensorflow for python

### 第 1 步 - 构建 Tensorflow（一次完成）

- 如果您已安装 Tensorflow CC 库和 Tensorflow CC 库，则执行第 2 步构建 OCM
- 设置 bazel，按照以下链接 https://www.tensorflow.org/install/source#install\_bazel 中提供的步骤操作
- 运行以下 bash 文件，它将构建 Tensorflow CC 和 Python 安装包，并在虚拟环境中安装 python 安装包  
`build_and_setup_tf.sh ${TF_SRC_DIR}`  
`# if the Tensorflow source code is already cloned at ${TF_SRC_DIR} then it will checkout the required version and will rebuild it`  
`# otherwise it will clone Tensorflow source code at ${TF_SRC_DIR} path and will build it`  
`# Note: building tensorflow could take several hours based on system config`

### 第 2 步 - 构建 OCM

- 前提条件 - Tensorflow 框架和 CC 库，如 libtensorflow\_framework.so 和 libtensorflow\_cc.so
- 如果已使用第 1 步构建和设置 tensorflow，则激活已创建的虚拟环境，
    `source ${TF_SRC_DIR}/ocm_venv/bin/activate`
- 否则，请确保当前的 python 环境已预安装所需版本的 Tensorflow，
    `build_ocm.sh ${TF_SRC_DIR}`  
`# make sure Tensorflow CC library(libtensorflow_cc.so) is present at this path ${TF_SRC_DIR}/tensorflow/bazel-bin/tensorflow/`

## 运行 OCM C++ 测试应用的步骤

`cd build`  
`./ov_ocm ${PATH_TO_TF_FROZEN_PB_FILE} ${DEVICE} {OPENVINO_VERSION}`

### 使用示例：

`./ov_ocm test.pb CPU 2021.2`  
`./ov_ocm test.pb MYRIAD 2021.1`