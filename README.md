### [中文](#中文) | [日本語](#日本語) | [English](#english)

---

<div lang="zh-CN">

## 中文

### Riichi Analysis Engine

一款兼容 [riichi-engine-protocol 2.1](https://github.com/SiyeW/riichi-engine-protocol) 的立直麻将分析引擎。

同一个模型可以提供动作推荐、对手向听、牌张放铳率、对手暗牌、牌山、对手宝牌与打点、小局结果与收支、终局顺位与分数预测。

### 当前状态

模型、数据转换和训练代码正在开发。仓库暂不提供训练数据和正式权重。

### 开发

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[train,test]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\check_repository.py
```

模型结构和训练方法见[模型文档](docs/model.zh-CN.md)，数据来源见[数据说明](DATA_PROVENANCE.md)。

### 许可证

源代码采用 [GNU Affero General Public License v3.0 or later](LICENSE)。训练数据、模型权重和第三方组件适用各自的许可条款。

</div>

---

<div lang="ja">

## 日本語

### Riichi Analysis Engine

[riichi-engine-protocol 2.1](https://github.com/SiyeW/riichi-engine-protocol) に対応するリーチ麻雀解析エンジンです。

1 つのモデルで、行動推薦、対戦相手のシャンテン数と放銃率、手牌と牌山、ドラ数と打点、局の結果と収支、最終順位と持ち点を予測します。

### 開発状況

モデル、データ変換、学習処理は現在開発中です。学習データと正式な重みはまだ公開していません。

### 開発

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[train,test]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\check_repository.py
```

モデル構成と学習方法は[モデル資料](docs/model.ja-JP.md)、データの出典は[データ資料](DATA_PROVENANCE.md)を参照してください。

### ライセンス

ソースコードは [GNU Affero General Public License v3.0 or later](LICENSE) で提供します。学習データ、モデルの重み、サードパーティー製コンポーネントには、それぞれのライセンスが適用されます。

</div>

---

<div lang="en">

## English

### Riichi Analysis Engine

A Riichi Mahjong analysis engine compatible with [riichi-engine-protocol 2.1](https://github.com/SiyeW/riichi-engine-protocol).

One model predicts recommended actions, opponent shanten and deal-in risk, concealed hands and the wall, dora and hand value, kyoku outcomes and score changes, and final placement and scores.

### Status

The model, data converter, and training pipeline are under development. Training data and release weights are not included yet.

### Development

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[train,test]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\check_repository.py
```

See the [model documentation](docs/model.en-US.md) for the architecture and training workflow, and [data provenance](DATA_PROVENANCE.md) for dataset sources.

### License

Source code is licensed under the [GNU Affero General Public License v3.0 or later](LICENSE). Training data, model weights, and third-party components retain their respective terms.

</div>
