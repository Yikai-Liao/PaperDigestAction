# PaperDigestAction

这是一个用于自动化论文摘要和推荐的 GitHub Action 项目。它旨在帮助您快速处理和管理学术论文，并根据您的个性化配置提供推荐。

## 环境配置

本仓库使用uv管理python 环境，你可以安装[uv](https://github.com/astral-sh/uv)后执行下面的命令来安装需要的依赖
```
uv python install 3.12
uv sync --frozen
```

## 如何 Fork 和个性化配置

要开始使用此项目并进行个性化配置，请按照以下步骤操作：

### 1. Fork 此仓库

点击 GitHub 页面右上角的 "Fork" 按钮，将此仓库复制到您的个人账户下。

### 2. 清除个性化配置

为了确保您从一个干净的状态开始，我们提供了一个 Python 脚本来清除所有示例和我的个性化配置。

运行以下命令来清除数据：

```bash
uv run python script/clear_personal_data.py
```

此脚本将执行以下操作：
*   删除 `preference/` 目录下的所有 `.csv` 文件。
*   删除 `summarized/` 目录下的所有 `.jsonl` 文件。

### 3. 个性化配置

您可以根据自己的需求修改以下文件：

#### `config.toml`

此文件包含项目的核心配置。您需要根据您的实际情况修改其中的值，特别是以下部分：

例如：
```toml
# config.toml 示例
[data]
categories = ["cs.CL", "cs.CV", "cs.AI", "cs.LG", "stat.ML", "cs.IR", "cs.CY"] # 替换为您感兴趣的论文类别
sample_rate = 0.004 # 调整论文采样率，以控制处理的论文数量
t_path = "./recommendations.parquet" # 推荐结果的输出路径，通常无需修改
```
请注意，此文件中不包含 GitHub 仓库信息和 API 密钥。这些敏感信息通常通过环境变量或 GitHub Actions Secrets 进行配置。

您还会看到 `[[llms]]` 部分，它定义了不同的 LLM 服务提供商及其配置。这些配置通常无需修改，除非您需要添加或调整特定的 LLM 服务。

#### `preference/` 目录

此目录用于存放您的个性化偏好数据（CSV 格式）。这些数据将用于训练推荐模型。

您可以创建类似 `preference/2025-05.csv` 的文件，其中包含您的论文偏好信息。

示例 `preference/init.csv`：
```csv
id,preference
2410.1285,like
2408.05646,dislike
```
`id` 是Arxiv论文的唯一标识符，`preference` 是您对论文的偏好（`like` 或 `dislike`）。

#### 生成偏好文件

为了方便您生成偏好文件，我们提供了一个脚本来从 Zotero 导出的 CSV 文件中提取 arXiv ID 并生成 `init.csv`。

**步骤 1: 从 Zotero 导出 CSV 文件**

1.  打开 Zotero 桌面应用。
2.  选择您想要导出的文献条目（例如，您可以选择一个集合或整个文献库）。
3.  右键点击选定的条目或集合，选择 `Export Items...`。
4.  在导出对话框中，选择 `CSV` 作为导出格式。
5.  确保勾选 `Export Files` 选项，以便包含文件附件路径（这有助于提取 arXiv ID）。
6.  点击 `OK` 并选择一个保存位置，例如 `test/data/zotero.csv`。

**步骤 2: 使用脚本生成偏好文件**

运行以下命令来生成偏好文件：

```bash
uv run python script/generate_preference.py test/data/zotero.csv
```

```bash
uv run python script/generate_preference.py test/data/zotero.csv -o preference/init.csv -p like
```

*   将 `test/data/zotero.csv` 替换为您实际导出的 Zotero CSV 文件路径。
*   `-o` 或 `--output_csv` 参数指定输出的偏好文件路径，默认为 `preference/init.csv`。
*   `-p` 或 `--preference` 参数指定所有提取的论文的偏好值，默认为 `like`。您可以将其设置为 `dislike` 或其他自定义值。

## GitHub Actions 设置

为了让项目自动化运行，您需要配置 GitHub Actions。

### 1. 启用 GitHub Actions

在您的 Fork 仓库中，导航到 "Actions" 选项卡。如果 Actions 被禁用，您可能需要点击 "I understand my workflows, go ahead and enable them." 来启用。

### 2. 权限设置

为了让 GitHub Actions 能够写入仓库内容（例如更新摘要文件），您需要授予其适当的权限。

导航到您的仓库设置：`Settings` -> `Actions` -> `General`。
在 "Workflow permissions" 部分，选择 "Read and write permissions"。

### 3. Secrets 设置

您的 OpenAI API 密钥等敏感信息不应直接暴露在代码中。您应该将其作为 GitHub Secrets 进行管理。

导航到您的仓库设置：`Settings` -> `Secrets and variables` -> `Actions`。
点击 "New repository secret" 并添加以下 Secrets。您需要根据您在 `config.toml` 中配置的 LLM 服务来设置相应的 API 密钥：

*   **Name**: `GH_TOKEN`
*   **Value**: 您的 Personal Access Token (PAT)。请确保此 PAT 具有 `repo` 和 `workflow` 权限，以便 GitHub Actions 可以读取和写入仓库内容，并执行工作流。**请妥善保管此 Token，并将其提供给您的 Telegram Bot。**

**生成 Personal Access Token (PAT)**

1.  导航到 GitHub 的 `Settings` -> `Developer settings` -> `Personal access tokens` -> `Tokens (classic)`。
2.  点击 `Generate new token` (classic)。
3.  为您的 Token 命名（例如：`PAPER_DIGEST_ACTION_PAT`）。
4.  设置过期时间（建议设置为 `No expiration` 或较长的期限，以便持续自动化）。
5.  在 `Select scopes` 部分，勾选以下权限：
    *   `repo` (所有子选项，包括 `repo:status`, `repo_deployment`, `public_repo`, `security_events`) - **这是必需的，以便 Actions 可以读取和写入仓库内容，并下载工作流生成的 artifact。**
    *   `workflow` - **这是必需的，以便 Actions 可以执行工作流。**
6.  点击 `Generate token`。
7.  **重要：** 复制生成的 Token。您将无法再次看到它。请妥善保管此 Token，并将其提供给您的 Telegram Bot。

*   **Name**: `XAI_API_KEY`
*   **Value**: 您的 Grok API 密钥 (如果使用 Grok 模型)

*   **Name**: `GEMINI_API_KEY`
*   **Value**: 您的 Gemini API 密钥 (如果使用 Gemini 模型)

*   **Name**: `DEEPSEEK_API_KEY`
*   **Value**: 您的 DeepSeek API 密钥 (如果使用 DeepSeek 模型)

*   **Name**: `DOUBAO_API_KEY`
*   **Value**: 您的 Doubao API 密钥 (如果使用 Doubao 模型)

请确保您使用的 LLM 服务对应的 API 密钥已正确配置为 GitHub Secret。

完成以上步骤后，您的 PaperDigestAction 项目就可以开始自动化运行了！