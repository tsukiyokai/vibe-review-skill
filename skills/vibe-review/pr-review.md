# PR/MR URL 处理流程

当参数是 PR/MR 链接时，按以下步骤执行。

## Step 1: 解析 URL

解析前先规范化 URL：移除 trailing slash、移除 `/files`、`/commits`、`/checks` 等 tab 页后缀、移除查询参数（`?` 及之后）、将 `http://` 统一为 `https://`。

从规范化后的 URL 中提取 PLATFORM、OWNER、REPO、PR_NUMBER：

| 平台    | URL 模式                                                 |
| ------- | -------------------------------------------------------- |
| GitHub  | `github.com/{OWNER}/{REPO}/pull/{PR_NUMBER}`             |
| GitCode | `gitcode.com/{OWNER}/{REPO}/pull/{PR_NUMBER}`            |
| GitLab  | `gitlab.com/{OWNER}/{REPO}/-/merge_requests/{PR_NUMBER}` |
| Gitee   | `gitee.com/{OWNER}/{REPO}/pulls/{PR_NUMBER}`             |

注意：OWNER 可能包含 `/`（如 GitLab 子群组 `group/subgroup`）。
仓库 HTTPS 地址：`https://{PLATFORM}/{OWNER}/{REPO}.git`

同时解析可选参数：
- `--base <branch>`：用户指定的目标分支（PR 合入的目标），优先级最高

## Step 2: 检查本地是否已有该仓库

1. 检查当前工作目录是否就是目标仓库（`git remote -v` 比对 OWNER/REPO）
2. 若不是，读取本 skill 目录下的 `references/standards-personal.md` 中的"本地仓库映射"表格，查找 `{PLATFORM}/{OWNER}/{REPO}` 对应的本地路径
3. 若找到匹配的本地仓库 → 在该仓库中 `git fetch origin` 后跳到 Step 4
4. 若均未命中 → 进入 Step 3

## Step 3: Clone 或复用仓库

```bash
WORK_DIR="/tmp/vibe-review-{REPO}-{PR_NUMBER}"
```

如果 `$WORK_DIR` 已存在且是有效 git 仓库（`git -C "$WORK_DIR" rev-parse --git-dir` 成功），则复用：
```bash
cd "$WORK_DIR"
git fetch origin
```

否则重新 clone：
```bash
rm -rf "$WORK_DIR"
git clone --filter=blob:none "https://{PLATFORM}/{OWNER}/{REPO}.git" "$WORK_DIR"
cd "$WORK_DIR"
```

关键说明：
- 使用 `--filter=blob:none`（blobless clone），保留完整 commit/tree 历史，仅按需下载 blob。确保 `git merge-base` 能正确工作
- 不使用 `--depth=N`（shallow clone），因为 shallow clone 截断历史会导致 `git merge-base` 失败或返回错误结果
- 如果 clone 失败，提示用户："git clone 失败，请确认 git credentials 已配置"

## Step 4: Fetch PR ref

**GitHub 平台优先尝试 gh api（静默降级）：**

如果平台是 GitHub，依次检查：
1. `which gh` — 未安装则跳过
2. `gh api repos/{OWNER}/{REPO}/pulls/{PR_NUMBER}` — 失败则跳过

若 `gh api` 成功，从返回 JSON 中提取：
- `base.ref` → TARGET_BRANCH（若用户未通过 `--base` 指定）
- `head.ref` → SOURCE_BRANCH
- 然后 `git fetch origin {SOURCE_BRANCH}:pr-{PR_NUMBER}` 并跳到 Step 5

**通用 ref fetch（所有平台的兜底方案）：**

按平台依次尝试不同的 PR ref 格式，首个成功即停止：

| 平台    | 优先尝试                          | 备选尝试                          |
| ------- | --------------------------------- | --------------------------------- |
| GitHub  | `pull/{PR_NUMBER}/head`           | --                                |
| GitCode | `merge-requests/{PR_NUMBER}/head` | `pull/{PR_NUMBER}/head`           |
| GitLab  | `merge-requests/{PR_NUMBER}/head` | --                                |
| Gitee   | `pull/{PR_NUMBER}/head`           | `merge-requests/{PR_NUMBER}/head` |
| 未知    | 依次尝试上述所有格式              | --                                |

```bash
git fetch origin {REF_FORMAT}:pr-{PR_NUMBER}
```

如果所有 ref 格式都 fetch 失败，告知用户：
"/vibe-review 无法获取 PR ref，请手动提供分支名：`/vibe-review {url} --base main --head feature-xxx`"

## Step 5: 生成 diff

确定目标分支（按优先级）：
1. 用户通过 `--base` 指定的分支
2. Step 4 中 `gh api` 返回的 `base.ref`
3. fallback：检查 `origin/main` 是否存在，存在则用 `main`，否则用 `master`

```bash
TARGET_BRANCH=<按上述优先级确定>

# 找到分叉点
MERGE_BASE=$(git merge-base origin/$TARGET_BRANCH pr-{PR_NUMBER})

# 生成 diff
git diff $MERGE_BASE pr-{PR_NUMBER}
```

如果 diff 超过 5000 行，告知用户变更量大，建议分批审查。

## Step 6: 执行 review 并提示清理

设定工作目录为仓库目录。仅在需要 Read/Grep 文件内容进行深度审查时，再 `git checkout pr-{PR_NUMBER}`。

按 SKILL.md 的"## 1. 理解变更上下文"开始正常审查流程。

review 完成后，如果使用的是临时 clone 目录，在报告末尾提示：
"临时仓库保留在 `{WORK_DIR}`，如需清理：`rm -rf {WORK_DIR}`"
