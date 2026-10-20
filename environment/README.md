# 运行环境说明

当前实测环境为 Windows、Python3.13.11，直接包版本见 [requirements-abc-tested.txt](requirements-abc-tested.txt)。训练采用 torch2.7.1+cu118 和 NVIDIA GeForce MX450；开发评估采用 CPU float32。

根目录 requirements.txt 给出宽松运行依赖范围，不包含完整工程测试环境。实测版本文件也不是通用系统锁文件：CUDA wheel、显卡驱动和 Python 版本需要相互兼容。PyYAML 提供完整 YAML 解析，pytest 用于工程验证。

实际安装完成后先核对包版本和CUDA可用性，再按[复现说明](../docs/reproduction/REPRODUCE_C_D.md)检查配置与测试。旧checkpoint的连续训练还依赖既有设备/数值路径，不能只凭包已安装就声称重放一致。

本次整理没有安装或升级依赖，没有修改已训练模型的运行环境。
