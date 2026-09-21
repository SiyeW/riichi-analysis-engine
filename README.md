# Riichi Analysis Engine

<p align="center">
  <img src="docs/assets/architecture-overview.webp" alt="Riichi Analysis Engine architecture overview">
</p>

### [中文](#中文) | [日本語](#日本語) | [English](#english)

---

<div lang="zh-CN">

## 中文

### Riichi Analysis Engine

一款兼容 [Riichi Engine Protocol 2.2](https://github.com/SiyeW/riichi-engine-protocol) 的立直麻将分析引擎。

同一个模型根据牌局信息提供动作推荐、对手向听、牌张放铳率、对手暗牌、牌山、对手宝牌与打点、小局结果与收支、终局顺位与分数预测。

### 分析输出

| 类别 | 输出 |
| --- | --- |
| 动作 | 候选动作与动作推荐 |
| 对手状态 | 三名对手的向听状态 |
| 放铳风险 | 各牌张对三名对手的放铳风险 |
| 对手暗牌 | 三名对手的手牌预测 |
| 牌山 | 剩余牌山预测 |
| 手牌价值 | 对手宝牌数量与打点预测 |
| 小局结果 | 小局结果与分数收支预测 |
| 整场结果 | 终局顺位与分数预测 |

这些输出由同一个模型提供，共用牌局状态表示和时序信息。

具体的输入编码、模型结构、输出头和通用训练接口见[模型文档](docs/model.zh-CN.md)。

### 与 Riichi Mahjong Studio 的关系

[Riichi Mahjong Studio](https://github.com/SiyeW/riichi-mahjong-studio) 是用于牌谱研究和对局练习的桌面程序，可以通过 Riichi Engine Protocol 加载外部引擎。

Riichi Analysis Engine 计划作为其中的分析引擎之一，将模型输出提供给 Studio 的牌局和研究界面。

引擎本身也可以由其他兼容 Riichi Engine Protocol 的程序调用。

### 当前状态

模型、数据转换和训练流程正在开发。

仓库目前包含：

- 引擎运行代码
- 模型结构与训练代码
- 数据转换工具
- 测试与仓库检查工具
- Windows 引擎包构建脚本
- 模型文档

目前不提供训练数据和正式模型权重。

### 开发

建议使用 Python 3.11。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[train,test,build]"
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

运行仓库检查：

```powershell
.\.venv\Scripts\python.exe scripts\check_repository.py
```

运行：

```powershell
.\build.ps1
```

可以生成不含模型权重的 Windows 引擎包。

### 文档

- [模型结构与通用训练接口](docs/model.zh-CN.md)
- [Riichi Engine Protocol](https://github.com/SiyeW/riichi-engine-protocol)
- [第三方组件声明](THIRD_PARTY_NOTICES.md)

### 许可证

源代码采用 [GNU Affero General Public License v3.0 or later](LICENSE)。

训练数据、模型权重和第三方组件适用各自的许可条款。

</div>

---

<div lang="ja">

## 日本語

### Riichi Analysis Engine

[Riichi Engine Protocol 2.2](https://github.com/SiyeW/riichi-engine-protocol) に対応するリーチ麻雀解析エンジンです。

1つのモデルから、行動推薦、対戦相手のシャンテン状態、牌ごとの放銃リスク、相手手牌、牌山、相手のドラ枚数と打点、局の結果と得失点、最終順位と持ち点を予測します。

### 解析出力

| 分類 | 出力 |
| --- | --- |
| 行動 | 候補行動と推奨行動 |
| 相手状態 | 3人の相手のシャンテン状態 |
| 放銃リスク | 各牌について3人の相手それぞれに対する放銃リスク |
| 相手手牌 | 3人の相手の手牌予測 |
| 牌山 | 残りの牌山の予測 |
| 手牌価値 | 相手のドラ枚数と打点の予測 |
| 局結果 | 局の結果と得失点の予測 |
| 半荘結果 | 最終順位と持ち点の予測 |

これらの出力は1つのモデルから生成され、局面表現と時系列情報を共有します。

入力エンコード、モデル構成、各出力ヘッド、共通の学習インターフェースについては[モデル資料](docs/model.ja-JP.md)を参照してください。

### Riichi Mahjong Studio との関係

[Riichi Mahjong Studio](https://github.com/SiyeW/riichi-mahjong-studio) は、牌譜研究と対局練習のためのデスクトップアプリケーションです。Riichi Engine Protocol を通じて外部エンジンを読み込めます。

Riichi Analysis Engine は、その解析エンジンの一つとして、モデルの出力を Studio の対局画面や研究画面に提供することを想定しています。

Riichi Engine Protocol に対応する他のプログラムから利用することもできます。

### 開発状況

モデル、データ変換、学習処理を現在開発しています。

リポジトリには以下が含まれています。

- エンジンの実行コード
- モデル構成と学習コード
- データ変換ツール
- テストとリポジトリチェック用ツール
- Windows エンジンパッケージのビルドスクリプト
- モデル資料

現在、学習データと正式なモデル重みは公開していません。

### 開発

Python 3.11 を推奨します。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[train,test,build]"
```

テスト：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

リポジトリチェック：

```powershell
.\.venv\Scripts\python.exe scripts\check_repository.py
```

以下を実行すると、

```powershell
.\build.ps1
```

モデル重みを含まない Windows エンジンパッケージを作成できます。

### ドキュメント

- [モデル構成と共通の学習インターフェース](docs/model.ja-JP.md)
- [Riichi Engine Protocol](https://github.com/SiyeW/riichi-engine-protocol)
- [サードパーティーコンポーネントに関する表記](THIRD_PARTY_NOTICES.md)

### ライセンス

ソースコードは [GNU Affero General Public License v3.0 or later](LICENSE) で提供します。

学習データ、モデル重み、サードパーティー製コンポーネントには、それぞれのライセンスが適用されます。

</div>

---

<div lang="en">

## English

### Riichi Analysis Engine

A Riichi Mahjong analysis engine compatible with [Riichi Engine Protocol 2.2](https://github.com/SiyeW/riichi-engine-protocol).

A single model provides action recommendations, opponent shanten estimates, tile-specific deal-in risk, opponent hand and wall predictions, opponent dora and hand-value estimates, kyoku outcome and score-change predictions, and final placement and score predictions.

### Analysis Outputs

| Category | Output |
| --- | --- |
| Actions | Candidate and recommended actions |
| Opponent state | Shanten state of each opponent |
| Deal-in risk | Tile-specific deal-in risk against each opponent |
| Opponent hands | Predicted concealed hands of the three opponents |
| Wall | Prediction of the remaining wall |
| Hand value | Opponent dora count and hand-value predictions |
| Kyoku result | Kyoku outcome and score-change predictions |
| Match result | Final placement and score predictions |

The outputs are produced by a single model and share the same game-state representation and temporal information.

See the [model documentation](docs/model.en-US.md) for input encoding, model architecture, output heads, and the common training interface.

### Riichi Mahjong Studio

[Riichi Mahjong Studio](https://github.com/SiyeW/riichi-mahjong-studio) is a desktop application for game-record study and game practice. It can load external engines through the Riichi Engine Protocol.

Riichi Analysis Engine is intended to provide its model outputs to Studio as one of its analysis engines.

It can also be used by other programs compatible with the Riichi Engine Protocol.

### Status

The model, data conversion, and training pipeline are under development.

The repository currently contains:

- engine runtime code
- model architecture and training code
- data-conversion tools
- tests and repository checks
- Windows engine-package build scripts
- model documentation

Training data and release model weights are not currently provided.

### Development

Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[train,test,build]"
```

Run the tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run the repository checks:

```powershell
.\.venv\Scripts\python.exe scripts\check_repository.py
```

Run:

```powershell
.\build.ps1
```

to create a Windows engine package without model weights.

### Documentation

- [Model architecture and common training interface](docs/model.en-US.md)
- [Riichi Engine Protocol](https://github.com/SiyeW/riichi-engine-protocol)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

### License

Source code is licensed under the [GNU Affero General Public License v3.0 or later](LICENSE).

Training data, model weights, and third-party components are subject to their respective terms.

</div>
