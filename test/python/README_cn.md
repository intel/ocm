Bash 脚本 run\_test\_setup，

- 下载带有算子单元测试代码的 tensorflow 仓库
- 下载 openvino\_tensorflow 单元测试运行程序并为该脚本安装补丁，以转储 .pb 文件（将输入更改为占位符之后）
- 创建python 虚拟环境，安装所有关联组件和补丁 tf test\_util.py，以禁用 eager 执行
- 构建用于推理的 OpenVINO benchmark 程序。
- 如要运行设置文件，请使用命令  
    `$./run_tests.sh ${OV_PATH}`


Bash 脚本 run\_test

- 运行 OCM、MO 和 INFER 作业。
- 在其他设备“CPU”、“GPU”、“MYRIAD”和 HDDL 上运行
- 运行单元测试运行程序，转储内部的 .pb 文件
- 通过 OCM 运行 .pb
- 通过 OpenVINO MO 运行 .pb 文件，以生成 IR 文件
- 通过benchmark应用，针对生成的 IR 运行推理

# 运行测试：

    `$./run_tests.sh <OV_PATH> <BUILD_TYPE> <MODE> <TEST_FILE> <DEVICE_TYPE> <MODEL_PATH>`

OV\_PATH 是 openvino 路径，TEST\_LIST 是单元测试列表。  
BUILD\_TYPE 可以是 OCM，MO，也可以是 INFER。可以用逗号隔开的形式给出一个或多个支持的选项，如 OCM,MO,INFER。  
MODE 可以是 UTEST（单元测试）或 MTEST（模型测试）。  
UNIT\_TEST\_FILE 列出了 TF 操作单元测试。该文件中的指定测试通过 bash 脚本执行。UTEST 模式的 run\_test.sh 需要  
MODEL\_TEST\_FILE 包含待测试的模型名称以及相关参数，如 input\_shape 等。MTEST 模式的 run\_test.sh 需要。

如何运行 OCM  
    \- 在模型上  
        `$./run_tests.sh OV_PATH OCM MTEST MODEL_TEST_FILE DEVICE_TYPE`  
    - 在单元测试上（也生成 PB 文件）
        `$./run_tests.sh OV_PATH OCM UTEST UNIT_TEST_FILE DEVICE_TYPE`  
    - 在预生成的 PB 文件的单元测试上  
	`$./run_tests.sh OV_PATH OCM UTEST UNIT_TEST_FILE DEVICE_TYPE ./pbfiles`

如何在模型上运行 MO  
    \- 在模型上  
        `$./run_tests.sh OV_PATH MO MTEST MODEL_TEST_FILE DEVICE_TYPE` 
    - 在单元测试 PB 文件上  
        `$./run_tests.sh OV_PATH MO UTEST UNIT_TEST_FILE DEVICE_TYPE ./pbfiles`

如何运行 INFER  
    \- 在模型上  
        `$./run_tests.sh OV_PATH INFER MTEST MODEL_TEST_FILE DEVICE_TYPE` 
    - 在单元测试 PB 文件上  
        `$./run_tests.sh OV_PATH INFER UTEST UNIT_TEST_FILE DEVICE_TYPE ./pbfiles`

执行测试后的输出目录：
- pbfiles：针对单元测试生成的 .pb 文件
- pbfiles\_mo：生成的模型优化器输出文件。
- tf\_ocm\_logs：OCM 日志
- tf\_mo\_logs：MO 执行日志
- tf\_infer\_logs：benchmark应用执行日志