# 【部门级】算子编码红线及TOPN问题

## 1编码红线

1.1 RED-01 除零保护：除法/取余操作必须进行除零保护
1.2 RED-02 数组越界保护：数组访问必须进行越界保护
1.3 RED-03 溢出翻转保护：加法、减法、乘法操作必须进行溢出和翻转保护
1.4 RED-04 变量初始化：变量使用前必须进行有效初始化
1.5 RED-05 指针空值保护：指针操作必须先赋值后访问并进行空指针保护
1.6 RED-06 资源申请释放匹配：申请资源（内存资源、文件句柄等）时使用和释放必须匹配
1.7 RED-07 避免data race: 避免data race

## 2 TOPN问题

2.1 TOP-01 特殊值与边界值处理：算子设计时必须要考虑nan/inf/-inf/±0等特殊值和其他边界值处理。
2.2 TOP-02 外部输入合法性校验：融合规则/Infershape/Tiling的外部输入使用时需要对合法性进行校验。
2.3 TOP-03 gm偏移使用int64: 涉及gm内存偏移或大小必须使用int64表示。
2.4 TOP-04 tiling_id语义不变：新增tiling_id时，已有tiling_id语义不能发生变化。
2.5 TOP-05 校验函数返回值：必须校验函数返回值。
2.6 TOP-06 atomic累加清零：atomic累加指令需将src(ub)与dst(gm)做清零处理。
2.7 TOP-07 属性从context获取：runtime2.0场景，属性需要从context中获取，不允许使用CompileInfo传递算子的属性值。
2.8 TOP-08 禁止整数转浮点计算：可整数计算时不允许转换成浮点数计算，必要时需要转换成精度更高的类型。
2.9 TOP-09 通信算子融合核间同步：涉及到和通信算子的融合时，多轮计算和集合通信之间需要增加核间同步。
2.10 TOP-10 Shape/Dtype获取方式：不要使用GetInputTensor获取Shape和Dtype，使用GetInputDesc获取的对象来获取Dtype，使用context获取对应的Shape。
2.11 TOP-11 局部变量指针生命周期：在生命周期内使用局部变量指针，避免野指针。
2.12 TOP-12 宏定义变量名冲突：宏定义中临时变量命名不能和外部变量冲突。
2.13 TOP-13 禁止dlopen管理的so使用thread_local: 由dlopen、dlclose管理的so，禁止使用thread_local声明变量。
