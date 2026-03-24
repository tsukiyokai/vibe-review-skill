# fix-false-positive Skill 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建 `fix-false-positive` skill，让 Claude 自动从 GitHub Issue 读取误报信息、修改 vibe-review skill 文件、验证修复、并创建 PR 供人工审核。

**Architecture:** 单一 skill 文件包含全部流程逻辑；通过 `gh api` 与 GitHub 交互；验证阶段内联执行 vibe-review 的 PR 审查流程（读取 `pr-review.md` 后按其步骤执行）。

**Tech Stack:** Claude Code skill（Markdown 指令文件）、GitHub CLI (`gh`)、git

---

## 文件结构

| 操作 | 路径 | 职责 |
|------|------|------|
| 新建 | `skills/fix-false-positive/SKILL.md` | skill 全部逻辑 |
| 新建 | `.github/ISSUE_TEMPLATE/false-positive.md` | 误报 issue 模板 |

---

### Task 1：创建 GitHub Issue 模板

**Files:**
- Create: `.github/ISSUE_TEMPLATE/false-positive.md`

- [ ] **Step 1: 创建 `.github/ISSUE_TEMPLATE/` 目录并写入 issue 模板**

```bash
mkdir -p .github/ISSUE_TEMPLATE
```

写入 `.github/ISSUE_TEMPLATE/false-positive.md`：

```markdown
---
name: 误报修复
about: 报告 vibe-review 的误报，由 Claude 自动修复
title: "误报：[规则ID] [简短描述]"
labels: false-positive
assignees: ''
---

## 误报信息

**PR 链接**：
<!-- 粘贴触发误报的 ops-transformer PR 链接，如 https://github.com/org/ops-transformer/pull/123 -->

**误报发现（粘贴审查报告原文）**：
<!-- 将 vibe-review 输出的完整发现块粘贴到此处，包含位置、规则、置信度等字段 -->

```
### #N [等级] 发现标题
- 位置：`文件:行号`
- 规则：规则编号
- 置信度：确定/较确定/待确认
（问题描述...）
```

**为什么是误报**：
<!-- 说明该发现不应被报出的原因，如：上游已判空、值域受限、RAII 管理等 -->
```

- [ ] **Step 2: 验证模板文件内容格式正确**

检查点：
- frontmatter 包含 `labels: false-positive`
- 三个必填字段均有清晰注释说明
- 示例代码块格式正确（嵌套 ``` 用缩进区分）

- [ ] **Step 3: commit**

```bash
git add .github/ISSUE_TEMPLATE/false-positive.md
git commit -m "feat: add false-positive issue template"
```

---

### Task 2：创建 fix-false-positive SKILL.md

**Files:**
- Create: `skills/fix-false-positive/SKILL.md`

这是本计划的核心交付物。skill 文件是给 Claude 读取并执行的 Markdown 指令，不是可编译的代码。

- [ ] **Step 1: 创建目录并写入 SKILL.md**

```bash
mkdir -p skills/fix-false-positive
```

写入 `skills/fix-false-positive/SKILL.md`，完整内容如下：

````markdown
---
name: fix-false-positive
description: "修复 vibe-review 的误报。从 GitHub Issue 读取误报信息，分析并修改 skill 规则文件，重新审查原 PR 验证修复，创建 PR 供人工审核。当用户提到'修复误报'、'fix false positive'、'处理误报 issue'时触发。"
argument-hint: "[issue-number...]"
allowed-tools: Read, Edit, Write, Grep, Glob, Bash(gh api *), Bash(gh issue *), Bash(gh pr *), Bash(git checkout *), Bash(git add *), Bash(git commit *), Bash(git push *), Bash(git diff*), Bash(git log*), Bash(git show*), Bash(git remote*), Bash(git fetch*), Bash(git clone*), Bash(git merge-base*), Bash(git rev-parse*), Bash(git branch*), Bash(wc *), Bash(rm -rf /tmp/vibe-review-*), Bash(ls /tmp/vibe-review-*), Bash(mkdir -p /tmp/vibe-review-*)
---

# 误报修复 Skill

## 参数解析

$ARGUMENTS

- 有参数（如 `42` 或 `42 43 44`）→ 处理指定 issue 编号
- 无参数 → 拉取当前仓库所有 label=`false-positive` 的 open issues

## 当前环境

- 仓库：!`git remote -v 2>/dev/null | head -1 || echo "NOT_A_GIT_REPO"`
- 工作目录：!`pwd`

---

## Step 1：获取待处理 Issue 列表

**有参数时**：直接使用参数中的 issue 编号列表。

**无参数时**：
```bash
gh api repos/{OWNER}/{REPO}/issues \
  --jq '[.[] | select(.state=="open" and (.labels[].name=="false-positive"))]'
```
从当前仓库 remote URL 解析 OWNER/REPO。若无 label=false-positive 的 open issue，输出"暂无待处理的误报 issue"后终止。

---

## Step 2：逐 issue 处理

对每个 issue 执行以下流程。多个 issue 串行处理。

### 2.1 解析 issue 内容

```bash
gh api repos/{OWNER}/{REPO}/issues/{NUMBER}
```

从 issue body 中提取：
- **PR URL**：`**PR 链接**：` 后的 URL
- **误报发现原文**：`**误报发现（粘贴审查报告原文）**：` 后的完整文本块（含规则 ID、位置、置信度）
- **误报理由**：`**为什么是误报**：` 后的文本

**解析失败处理**：若上述任意字段缺失，在终端输出警告（`⚠️ Issue #N 缺少必填字段 [字段名]，跳过`），记录跳过原因，继续处理下一个 issue。

从误报发现原文中自动解析：
- 规则 ID（如 `红线1.5`、`2.18.1`）
- 代码位置（如 `src/ops/relu.cpp:45`）
- 发现等级（严重/一般/建议）

### 2.2 阅读现有规则

读取以下文件，理解当前规则上下文：
- `skills/vibe-review/references/false-positives.md`：已知误报模式
- `skills/vibe-review/references/standards-project-ops-transformer.md`：项目专项规则
- `skills/vibe-review/SKILL.md`：审查主流程（重点看与该规则 ID 相关的部分）

### 2.3 判断修改位置并执行修改

根据误报性质自主判断：

**情况 A：已知的代码模式例外**（如特定框架保证非空、特定类型不会溢出）
→ 追加条目到 `skills/vibe-review/references/false-positives.md`

追加规则：
- 在文件中找到与该规则对应的分类（空指针类/整数溢出类/资源泄漏类等）
- 在该分类下追加新条目，格式与已有条目一致：
  ```
  - [模式描述]：[为什么是误报]（需工具验证的条件）
  ```
- 若无匹配分类，在文件末尾新建分类并追加

**情况 B：规则本身覆盖太宽**（规则描述未排除某类合法场景）
→ 修改 `skills/vibe-review/references/standards-project-ops-transformer.md` 中的规则描述，在"排除"或"注意"子项中补充例外条件

→ 仅当问题在通用规则层（非项目专项）时，修改 `skills/vibe-review/SKILL.md` 中对应规则的排除说明

### 2.4 验证循环（最多3轮）

**每轮执行：**

1. 清理临时目录（确保干净工作区）：
   ```bash
   REPO_NAME=$(echo {PR_URL} | sed 's|.*/\([^/]*\)/pull/.*|\1|')
   PR_NUM=$(echo {PR_URL} | sed 's|.*/pull/\([0-9]*\).*|\1|')
   rm -rf /tmp/vibe-review-${REPO_NAME}-${PR_NUM}
   ```

2. 对 issue 里的 PR 重新执行 vibe-review 的 PR 审查流程：
   读取 `skills/vibe-review/pr-review.md`，按其中步骤完整执行对该 PR 的审查，生成新的审查报告。

3. **判定**：对比新报告与原误报发现，判断是否仍存在"相同根因"的发现：
   - 相同根因定义：规则 ID 相同 + 同一代码位置（允许行号因代码变更有小幅偏移）
   - **通过条件**：新报告中该发现已消失，或出现但置信度已降为"待确认"
   - **失败条件**：新报告中该发现仍以"确定"或"较确定"出现

4. **通过** → 执行：
   ```bash
   git add skills/vibe-review/references/false-positives.md \
           skills/vibe-review/references/standards-project-ops-transformer.md \
           skills/vibe-review/SKILL.md
   git commit -m "fix: 修复误报 #N（规则 {RULE_ID}，{LOCATION}）"
   ```
   退出循环，进入 2.5。

5. **失败且未达3轮** → 重新分析为何修改未能生效，调整修改方案，回到步骤 1（勿 commit，保持文件已修改未暂存状态）。

6. **第3轮仍失败** → 回滚：
   ```bash
   git checkout -- skills/vibe-review/references/false-positives.md \
                   skills/vibe-review/references/standards-project-ops-transformer.md \
                   skills/vibe-review/SKILL.md
   ```
   在原 issue 下发布失败报告（见 Step 3 失败格式），标记该 issue 为失败，继续处理下一个 issue。

### 2.5 记录验证报告

将验证结果（包含通过轮次、修改位置、修改内容、新审查报告全文）暂存到内存，在 Step 4 中统一发布为 PR comment。

---

## Step 3：汇总处理结果

统计：通过 issue 列表、失败 issue 列表。

**若所有 issue 均失败**：
- 失败报告已在各 issue 下发布（见 2.4 步骤6）
- 在终端输出汇总："所有 issue 均修复失败，无可提交的更改。"
- 终止，不创建 PR。

**若至少有一个 issue 通过**：进入 Step 4。

---

## Step 4：创建 PR

### 4.1 推送分支

```bash
ISSUE_NUMS={通过的 issue 编号列表，用-连接}
git checkout -b fix/false-positive-issues-${ISSUE_NUMS}
git push origin fix/false-positive-issues-${ISSUE_NUMS}
```

注意：若处理多个 issue 时已在该分支上逐个 commit，直接 push 即可；若当前在 main 分支上已有 commit，需先创建分支再 push。

### 4.2 创建 PR

```bash
gh pr create \
  --title "fix: 修复误报 #{通过的issue编号列表}" \
  --body "$(cat <<'EOF'
## 误报修复

本 PR 由 fix-false-positive skill 自动生成。

### 处理结果

| Issue | 结论 | 修改位置 |
|-------|------|----------|
| #{N} | ✅ 通过（第X轮） | false-positives.md |
| #{M} | ❌ 失败（已在 issue 下报告） | — |

{通过的 issue 用 Closes #N，失败的用 Related #N}

> 验证报告见各 issue 对应的 PR comment。
EOF
)"
```

### 4.3 发布验证报告

对每个处理过的 issue（通过和失败均发布），向 PR 发布独立 comment：

**通过（第1轮）**：
```
## 验证报告：#{N}

**误报**：`{位置}` — {描述}（{规则ID}）
**修改位置**：`{文件路径}`
**修改内容**：{一句话说明}

**验证结果**：✅ 通过（第1轮）

重新审查 PR #{PR编号} 后，该发现未出现在报告中。

<details>
<summary>完整审查报告</summary>

{新审查报告全文}

</details>
```

**通过（多轮）**：在验证结果行后追加每轮的失败原因和调整说明。

**失败**：
```
## 验证报告：#{N}

**误报**：`{位置}` — {描述}（{规则ID}）

**验证结果**：❌ 失败（已迭代3轮，无法自动修复）

已回滚本 issue 的所有修改，请人工分析处理。

| 轮次 | 修改方案 | 失败原因 |
|------|----------|----------|
| 第1轮 | {方案描述} | 发现仍出现 |
| 第2轮 | {方案描述} | 发现仍出现 |
| 第3轮 | {方案描述} | 发现仍出现 |
```

**失败的 issue 还需额外在原 issue 下发布同一失败报告**：
```bash
gh issue comment {N} --body "{失败报告内容}"
```

---

## 约束提示

- 不修改 `skills/vibe-review/references/google-cpp-style-guide.md` 等外部引用文件
- 不修改 `skills/vibe-review/pr-review.md`（PR 审查流程本身）
- 每个 issue 的修改在验证通过前不 commit，避免污染 git history
- 串行处理 issue，避免对同一文件的并发修改冲突
````

- [ ] **Step 2: 检查 SKILL.md 文件**

检查点：
- frontmatter 格式正确（`---` 开头结尾，字段名无拼写错误）
- `allowed-tools` 单行逗号分隔（与 vibe-review/SKILL.md 格式一致）
- `$ARGUMENTS` 占位符存在
- `!`` `` 命令行用反引号包裹（环境变量注入语法）
- 所有流程步骤均有足够细节，无 "TODO" 或 "待补充" 占位符

- [ ] **Step 3: commit**

```bash
git add skills/fix-false-positive/SKILL.md
git commit -m "feat: add fix-false-positive skill"
```

---

### Task 3：手动验证 Skill 可被 Claude Code 加载

**Files:**
- Read: `skills/fix-false-positive/SKILL.md`
- Read: `.github/ISSUE_TEMPLATE/false-positive.md`

- [ ] **Step 1: 验证 frontmatter 可被解析**

```bash
# 检查 SKILL.md frontmatter 格式（--- 分隔符存在，name/description/allowed-tools 字段存在）
head -10 skills/fix-false-positive/SKILL.md
```

预期输出：第1行 `---`，第2行 `name: fix-false-positive`，第6行左右出现第二个 `---`。

- [ ] **Step 2: 验证 allowed-tools 覆盖 vibe-review 所需的全部工具**

```bash
# 提取 vibe-review 的 allowed-tools
grep "^allowed-tools:" skills/vibe-review/SKILL.md

# 提取 fix-false-positive 的 allowed-tools
grep "^allowed-tools:" skills/fix-false-positive/SKILL.md
```

逐一确认 vibe-review 中的每个工具都在 fix-false-positive 的列表中出现，另外还有 `Edit`, `Write`, `Glob`, `Bash(git add*)`, `Bash(git commit*)`, `Bash(git push*)`, `Bash(gh issue*)`, `Bash(gh pr*)` 等写操作工具。

- [ ] **Step 3: 验证 issue 模板 labels 字段**

```bash
head -8 .github/ISSUE_TEMPLATE/false-positive.md
```

预期：frontmatter 中包含 `labels: false-positive`。

- [ ] **Step 4: 最终 commit（如有遗漏修正）**

```bash
git add -p  # 仅 stage 有需要的修正
git commit -m "fix: correct skill frontmatter/template issues"
```

若无修正，跳过此步。
