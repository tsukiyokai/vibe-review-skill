---
name: fix-false-positive
description: "修复 vibe-review 的误报。从 GitHub Issue 读取误报信息，分析并修改 skill 规则文件，重新审查原 PR 验证修复，创建 PR 供人工审核。当用户提到'修复误报'、'fix false positive'、'处理误报 issue'时触发。"
argument-hint: "[issue-number...]"
allowed-tools: Read, Grep, Glob, Edit, Write, Bash(git diff*), Bash(git log*), Bash(git show*), Bash(git remote*), Bash(git clone*), Bash(git fetch*), Bash(git checkout*), Bash(git merge-base*), Bash(git rev-parse*), Bash(git branch*), Bash(git add*), Bash(git commit*), Bash(git push*), Bash(gh api *), Bash(gh issue *), Bash(gh pr *), Bash(wc *), Bash(rm -rf /tmp/vibe-review-*), Bash(ls /tmp/vibe-review-*), Bash(mkdir -p /tmp/vibe-review-*)
---

# 误报修复 Skill

## 参数解析

$ARGUMENTS

- 有参数（如 `123` 或 `123 456`）：将空格或逗号分隔的数字解析为 issue 编号列表
- 无参数：自动查询带 `false-positive` 标签的开放 issue

## 当前环境

- 仓库：!`git remote -v 2>/dev/null | head -1 || echo "NOT_A_GIT_REPO"`
- 工作目录：!`pwd`

---

## Step 1：获取待处理 Issue 列表

从上方环境信息中解析 OWNER 和 REPO（取 `git remote -v` 第一行的仓库路径）。

**有参数时：** 直接使用参数中的 issue 编号列表。

**无参数时：** 查询带 `false-positive` 标签的开放 issue：

```bash
gh api repos/{OWNER}/{REPO}/issues --jq '[.[] | select(.state=="open" and (.labels[].name=="false-positive"))]'
```

若未找到任何 issue，输出"未找到待处理的误报 issue"后退出。

创建工作分支：
在处理任何 issue 之前，先创建功能分支并切换到该分支，避免直接在主分支上提交：
```bash
BRANCH_NAME=fix/false-positive-issues-$(date +%Y%m%d-%H%M%S)
git checkout -b $BRANCH_NAME
```
分支名称最终将在 Step 4 中根据实际通过的 issue 编号重命名，此处先创建临时分支用于隔离工作。

---

## Step 2：逐 issue 处理

按 issue 编号升序逐条处理。记录每条处理结果（通过/失败），用于 Step 3 汇总和 Step 4 建 PR。

### 2.1 解析 issue 内容

读取 issue 正文：`gh api repos/{OWNER}/{REPO}/issues/{N}`

从正文中提取以下必填字段：
- **PR 链接**：`**PR 链接**：` 之后的内容
- **误报原文**：`**误报发现（粘贴审查报告原文）**：` 之后的内容
- **误报原因**：`**为什么是误报**：` 之后的内容

任一字段缺失时，输出 `⚠️ Issue #N 缺少必填字段 [字段名]，跳过`，继续处理下一条 issue。

从误报原文中自动解析：
- **rule_id**：形如 `规则：2.18.1` 或 `红线1.5` 的规则编号
- **location**：形如 `位置：src/foo.cpp:142` 的代码位置（文件路径 + 行号）
- **severity**：`[严重]` / `[一般]` / `[建议]`

### 2.2 阅读现有规则

读取以下文件，理解现有误报模式和规则定义：

- `skills/vibe-review/references/false-positives.md` — 已知误报模式
- 项目级规则文件（按需读相关章节）：根据 PR URL 中的仓库名选择对应的项目规则文件（与 vibe-review/SKILL.md 中的按仓库加载规则保持一致）：
  - ops-transformer → `standards-project-ops-transformer.md`
  - ops-nn → `standards-project-ops-nn.md`
  - hccl / hcomm → `standards-project-hccl.md`
  - 其他仓库 → 无额外项目规则文件

  若对应的项目规则文件存在，读取它；若不存在，仅读取 false-positives.md 和 SKILL.md。
- `skills/vibe-review/SKILL.md` — 仅阅读与该发现相关的规则章节（用 Grep 定位 rule_id）

同时用 Grep 在仓库中搜索该代码位置，理解其真实语义。

### 2.3 判断修改位置并执行修改

根据误报原因，判断修复属于哪种情况：

**Case A — 已知代码模式例外**（规则本身正确，但该代码模式不适用）

修改 `skills/vibe-review/references/false-positives.md`：
- 找到最匹配的分类（空指针类/整数溢出类/资源泄漏类/敏感信息类等）
- 在该分类下追加一条误报模式描述，说明该模式为何是误报
- 若无合适分类，在文件末尾新建分类

**Case B — 规则本身过宽**（规则描述需要收窄条件）

修改 `skills/vibe-review/references/standards-project-ops-transformer.md`（如果规则在项目级文件中）：
- 精确定位规则条目，在描述中补充排除条件或澄清适用范围
- 仅当问题出现在通用规则层时才修改 `skills/vibe-review/SKILL.md`

**禁止修改：** `skills/vibe-review/references/google-cpp-style-guide.md` 及其他外部参考文件、`skills/vibe-review/pr-review.md`。

### 2.4 验证循环（最多 3 轮）

**每轮步骤：**

1. 清理临时目录：
   ```bash
   rm -rf /tmp/vibe-review-{REPO}-{PR_NUM}
   ```

2. 重新执行 PR 审查：读取 `skills/vibe-review/pr-review.md`，严格按其流程对 issue 中的 PR URL 执行完整审查。审查时使用修改后的规则文件（当前工作目录中的版本）。

3. 判断结果（"同一根因"定义：rule_id 相同 且 代码文件路径相同，允许行号偏移）：
   - **通过（PASS）**：该发现未出现在报告中，或已降级为"待确认"
   - **失败（FAIL）**：该发现仍以"确定"或"较确定"出现在报告中

4. **PASS 时：**
   ```bash
   git add <修改的文件>
   git commit -m "fix: 修复误报 #N（规则 {RULE_ID}，{LOCATION}）"
   ```
   退出验证循环，进入下一条 issue。

5. **FAIL 且未到第 3 轮：** 分析本轮失败原因，调整修改方案，重新执行修改后进入下一轮。**本轮不提交。**

6. **FAIL 且已是第 3 轮：**
   - 回滚本 issue 的所有修改（仅回滚本轮实际修改的文件）：
     ```bash
     git checkout -- skills/vibe-review/references/false-positives.md \
                     skills/vibe-review/references/standards-project-ops-transformer.md \
                     skills/vibe-review/references/standards-project-ops-nn.md \
                     skills/vibe-review/references/standards-project-hccl.md \
                     skills/vibe-review/SKILL.md 2>/dev/null || true
     ```
     （`2>/dev/null || true` 处理文件未被修改的情况）
   - 在原始 issue 上发布失败注释：
     ```bash
     gh issue comment {N} --body "..."
     ```
     注释内容包含迭代历史（见 Step 4 FAIL 格式）。
   - 将该 issue 记录为失败，继续处理下一条。

**约束：** 轮次间不提交；每轮必须清理临时目录后重新 review。

### 2.5 记录验证报告

每条 issue 处理完成后，记录以下信息用于 Step 4：
- issue 编号、PR URL、PR 编号
- rule_id、location、severity
- 修改的文件名和修改内容摘要（一行）
- 最终结果：PASS（第几轮）或 FAIL（含各轮失败原因）
- 完整的最终审查报告（PASS 时）

---

## Step 3：汇总处理结果

统计通过和失败的 issue 数量并打印摘要：

```
处理完成：
✅ 通过：#N1, #N2（已提交）
❌ 失败：#N3（已回滚，已在 issue 上留言）
```

- 若**所有 issue 均失败**：打印摘要后退出，不创建 PR。同时清理临时工作分支：
  ```bash
  git checkout main
  git branch -d $BRANCH_NAME
  ```
- 若**至少有一条通过**：继续执行 Step 4。

---

## Step 4：创建 PR

### 4.1 推送分支

将通过的 issue 编号用 `-` 连接：

```bash
# 将工作分支重命名为最终分支名
ISSUE_NUMS={通过的 issue 编号，用-连接}
git branch -m fix/false-positive-issues-${ISSUE_NUMS}
git push origin fix/false-positive-issues-${ISSUE_NUMS}
```

注意：各 issue 的提交已在 Step 2.4 完成，直接在当前分支上推送即可。

### 4.2 创建 PR

使用 `gh pr create` 创建 PR：

- **标题**：`fix: 修复误报 #{通过的issue编号，空格分隔}`
- **Body**：

```
## 处理汇总

| Issue | 误报位置 | 规则 | 修改文件 | 结果 |
|-------|----------|------|----------|------|
| #N1   | location | rule | file     | ✅ 通过 |
| #N2   | location | rule | file     | ❌ 失败（已回滚） |

{对每个通过的 issue}
Closes #N1

{对每个失败的 issue（若在同一 PR 中提及）}
Related #N2
```

### 4.3 发布验证报告

对每条已处理的 issue（无论 PASS 还是 FAIL），在 PR 上发布一条验证报告注释。

**PASS 格式（单轮）：**

```
gh pr comment {FIX_PR_URL} --body "$(cat <<'EOF'
## 验证报告：#{N}
**误报**：`{location}` — {description}（{rule_id}）
**修改位置**：`{file}`
**修改内容**：{一行摘要}
**验证结果**：✅ 通过（第1轮）
重新审查 PR #{pr_num} 后，该发现未出现在报告中。
<details><summary>完整审查报告</summary>

{完整审查报告}

</details>
EOF
)"
```

**PASS 格式（多轮）：** 在"验证结果"前追加各轮失败原因：

```
**第1轮失败原因**：{原因}
**第2轮失败原因**：{原因}
**验证结果**：✅ 通过（第3轮）
```

**FAIL 格式：**

```
gh pr comment {FIX_PR_URL} --body "$(cat <<'EOF'
## 验证报告：#{N}
**误报**：`{location}` — {description}（{rule_id}）
**验证结果**：❌ 失败（已迭代3轮，无法自动修复）
已回滚本 issue 的所有修改，请人工分析处理。

| 轮次 | 修改方案 | 失败原因 |
|------|----------|----------|
| 第1轮 | {方案} | {失败原因} |
| 第2轮 | {方案} | {失败原因} |
| 第3轮 | {方案} | {失败原因} |
EOF
)"
```

失败的 issue 在 Step 2.4 步骤 6 已通过 `gh issue comment` 在原始 issue 上留言，此处在 PR（`{FIX_PR_URL}` 即 Step 4.2 创建的修复 PR）上再追加一条汇总。

---

## 约束提示

- 不修改 `google-cpp-style-guide.md` 或其他外部参考文件
- 不修改 `pr-review.md`（PR 审查流程本身）
- 验证轮次之间不提交（仅 PASS 后提交）
- 串行处理各 issue，不并行
- 每轮验证前必须清理 `/tmp/vibe-review-{REPO}-{PR_NUM}` 以确保全量重新 clone
