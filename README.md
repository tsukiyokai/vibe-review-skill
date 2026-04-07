# @tsukiyokai/vibe-review

[![npm](https://img.shields.io/npm/v/@tsukiyokai/vibe-review)](https://www.npmjs.com/package/@tsukiyokai/vibe-review)

[Claude Code](https://docs.anthropic.com/en/docs/claude-code)的代码审查skill —— 结构化、置信度分级、标准感知。支持C++和Python。

## 快速开始

```bash
npx @tsukiyokai/vibe-review --global
```

或通过 [skills.sh](https://skills.sh/) 安装：

```bash
npx skills add cann-ai-code-reviewer/vibe-review-skill
```

安装后在Claude Code中即可使用：

```
> /vibe-review src/transport.cpp                      # 单文件审查
> /vibe-review HEAD~3..HEAD                           # git range审查
> /vibe-review https://github.com/org/repo/pull/42    # PR审查
```

## 核心能力

- 先读完整上下文（调用者、头文件、基类），再做判断
- 用工具验证每个发现 —— 不猜行号，不猜指针是否为空
- 约束反证：对每个非"确定"发现，主动构造反证条件并用工具验证，抑制误报
- 标注置信度（确定 / 较确定 / 待确认）和严重级别（严重 / 一般 / 建议）
- 引用具体规则编号，过滤已知误报模式

## 工作流程

1. 解析目标 — 文件 / git range / PR URL
2. 加载分层标准
3. 读取完整上下文 — grep调用者、头文件、基类
4. 逐条检查必检规则 — 空指针、越界、格式串…
5. 工具验证每个发现
6. 约束反证 — 对非"确定"发现执行反证检验（见下方说明）
7. 输出结构化报告

## 分层标准

标准按层级加载 —— 外层是通用规则，内层是团队和项目的细化。skill自动识别当前仓库，加载匹配的标准层。

```
┌─────────────────────────────────┐
│ Company                         │  standards-company.md
│ ┌─────────────────────────────┐ │
│ │ Department                  │ │  standards-dept.md
│ │ ┌─────────────────────────┐ │ │
│ │ │ Product Line            │ │ │  standards-productline-*.md
│ │ │ ┌─────────────────────┐ │ │ │
│ │ │ │ Project             │ │ │ │  standards-project-*.md
│ │ │ │ ┌─────────────────┐ │ │ │ │
│ │ │ │ │ Personal        │ │ │ │ │  standards-personal.md
│ │ │ │ └─────────────────┘ │ │ │ │
│ │ │ └─────────────────────┘ │ │ │
│ │ └─────────────────────────┘ │ │
│ └─────────────────────────────┘ │
└─────────────────────────────────┘
```

| 层级         | 文件                         | 内容                    |
| ------------ | ---------------------------- | ----------------------- |
| Company      | `standards-company.md`       | 公司级规范              |
| Department   | `standards-dept.md`          | 部门红线 & TOP-N        |
| Product Line | `standards-productline-*.md` | 产品线规范(CANN C++)    |
| Project      | `standards-project-*.md`     | 项目级规范(HCCL/FA/MC2) |
| Personal     | `standards-personal.md`      | 个人审查习惯            |

## 约束反证

AI代码检视的核心挑战是误报。纯LLM判断"这像是个bug"不够可靠 —— 必须反过来问：在什么条件下这个问题不成立？然后用工具验证。

受腾讯LLM4PFA（[arxiv 2506.10322](https://arxiv.org/abs/2506.10322)）启发，vibe-review对每个非"确定"置信度的发现执行约束反证：

1. 列出反证条件 — 按bug类别枚举"若成立则为误报"的条件
2. 工具验证 — 用Read/Grep/git show验证每个条件，不凭推测
3. 判定 — 反证成立则抑制；部分成立则标"待确认"并注明原因；不成立则升级为"较确定"并附依据

典型反证条件：

| bug类别      | 反证条件（成立则为误报）                             |
| ------------ | ---------------------------------------------------- |
| 空指针解引用 | 调用链上游已判空；构造函数保证非空；值域约束排除null |
| 整数溢出     | 操作数受业务约束（枚举值、小常量）；类型提升后安全   |
| 资源泄漏     | RAII/智能指针管理；析构函数自动释放                  |
| 数组越界     | 索引受循环条件/前置校验约束；容器有边界保护          |
| 并发问题     | 变量仅单线程访问；已有锁/原子操作保护                |

LLM4PFA在工业级C/C++项目(Linux Kernel、OpenSSL、Libav)上过滤了72%-96%的误报，同时保持0.93的recall。vibe-review采用类似的"先提取约束、再验证可行性"思路，将其适配到交互式code review场景。

### 逻辑结构

正向思维是归纳式的："这里有个指针解引用，没看到判空，所以可能是bug。"这种推理链很脆弱，因为"没看到"不等于"不存在"，LLM的上下文窗口只是代码库的一个切面。反证把推理方向翻转：不问"这是不是bug"，而问"在什么条件下这不是bug"，然后尝试验证这些条件是否成立。

```
Let P = "defect X exists in this code"

Refutation:
  1. enumerate {C1, C2, ..., Cn} where Ci => not-P
  2. verify each Ci with tools
  3. any Ci = true  =>  not-P holds   =>  suppress
     all Ci = false =>  P reinforced  =>  report with evidence
     Ci uncertain   =>  undetermined  =>  mark "needs confirm"
```

这本质上是Popperian falsification在代码审查中的应用：一个发现的可信度不在于有多少证据支持它，而在于它经历了多少次否证尝试仍然存活。

每类bug的反证条件对应该类缺陷的一种"合法豁免机制"（空指针->上游判空/工厂保证非空/值域排除null；整数溢出->值域约束/类型提升；资源泄漏->RAII/智能指针/析构释放...）。经验丰富的reviewer在脑中隐式做这件事；此skill把隐性知识显式化，让LLM也能系统地执行。`false-positives.md`固化了已确认的豁免模式（如RDMA rkey不是敏感信息、循环计数器i++不会溢出），是一个可增量维护的反证知识库。

整体形成三层筛选管道：

```
┌────────────────────┐
│  Mandatory rules   │  high recall
└──────────┬─────────┘
           ▼
┌────────────────────┐
│  Tool verification │  upgrade confidence
└──────────┬─────────┘
           ▼
┌────────────────────┐
│  Refutation        │  suppress FP
└──────────┬─────────┘
           ▼
      [ Output ]
```

每层做精度和召回率的trade-off：前面宁多勿漏（高召回），后面逐步过筛（提高精度）。最终的置信度标签（确定/较确定/待确认）直接映射了发现通过了多少层验证。

### 局限性与诚实评估

这套方法有两个结构性缺陷，使用者应当了解。

其一，单体反证是伪对抗。反证的认识论前提是对抗性：提出命题的人和试图否证的人之间存在真实张力。法庭上控方和辩方是不同的人，formal verification里prover和verifier是不同的系统。此skill中，提出"这是bug"和试图否证"这是bug"的是同一个LLM、同一个上下文窗口、同一轮推理，不存在真实对抗。LLM有内在的coherence bias，倾向于让推理链自洽，而非adversarially attack自己的结论。当LLM"确信"是bug时，列出的反证条件往往是容易驳回的稻草人；当LLM不确定时，又会过度热情地接受反证条件，把真实缺陷抑制掉。

去掉"反证"的认识论包装，把Section 4.5改写成一个checklist："报空指针前先grep调用者、报溢出前先查值域、报泄漏前先查RAII"，效果是一样的。形式化反证框架没有提供超出checklist的额外推理能力。真正降低误报的机制是框架中嵌入的那些grep/read指令，不是反证的逻辑形式。

其二，结构性不对称。反证只作用于findings，不作用于non-findings。它过滤假阳性，但对假阴性完全无能为力。没有一个对称的步骤说"对于你没报出的每类bug，反证'这里没有bug'这个命题"。Section 3的必检规则部分补偿了这一点，但那是正向枚举，不是反证。一个完备的形式化方法应该对两个方向都有约束力。

这是否意味着约束反证无用？不是。它的实际价值在于：(1) 迫使LLM在报出发现前执行具体的工具调用，这是真正降低误报的机制；(2) 用"反证"框架把隐性验证步骤编码为显式流程，对prompt engineering有效；(3) 生成的反证文本给人类reviewer提供了判断依据。但应诚实地认识到：它是一个结构化checklist穿了形式化推理的外衣。

未来改进方向：dual-agent架构（一个agent报bug，另一个独立agent做refutation），或把refutation拆到独立的prompt pass里，制造真正的信息隔离和对抗张力。

## 输出示例

````
## 变更概述

本MR为TcpTransport添加重试逻辑，涉及2个文件，新增45行。

## 审查发现

共发现2个问题（严重2 / 一般0 / 建议0）

---

### #1 [严重] 使用了禁用函数memcpy
- 位置：src/transport/tcp_transport.cpp:142
- 规则：2.18.1
- 置信度：确定

问题代码：
```cpp
memcpy(dst, src, len);
```

修复建议：
```cpp
errno_t ret = memcpy_s(dst, dstMax, src, len);
CHK_SAFETY_FUNC_RET(ret);
```

---

### #2 [严重] Destroy持锁期间执行阻塞同步操作
- 位置：src/channel/hcomm_channel.cpp:245-250
- 规则：CON-04
- 置信度：较确定
- 反证：已检查是否有异步替代路径或锁外执行的可能。确认hcclStreamSynchronize必须在channelMutex_持锁期间调用（stream句柄在锁保护范围内），且无超时参数。反证条件不成立。

问题代码：
```cpp
std::lock_guard<std::mutex> lock(channelMutex_);
// ... 持锁期间
hcclStreamSynchronize(stream);  // 可阻塞60s+
```

修复建议：
将stream同步移到锁外，或使用带超时的同步API。

---

## 总结

建议优先处理2个严重问题。整体重试逻辑合理。
````

## 安装 / 卸载

```bash
# 方式一：npm
npx @tsukiyokai/vibe-review --global             # 全局安装
npx @tsukiyokai/vibe-review                      # 项目级安装
npx @tsukiyokai/vibe-review --remove --global    # 全局卸载
npx @tsukiyokai/vibe-review --remove             # 项目级卸载

# 方式二：skills.sh
npx skills add cann-ai-code-reviewer/vibe-review-skill      # 安装
npx skills rm --global vibe-review               # 卸载
```

## 自定义

安装后编辑 `references/` 下的标准文件：

| 文件                    | 用途                |
| ----------------------- | ------------------- |
| `standards-company.md`  | 公司级编码规范      |
| `standards-dept.md`     | 部门红线及TOP-N问题 |
| `standards-personal.md` | 个人审查偏好        |
| `false-positives.md`    | 需要抑制的误报模式  |

如需支持新项目，创建 `standards-project-<name>.md` 并更新 `SKILL.md` 中的路由表。
