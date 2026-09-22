# SAO Extraction Module — `backend/sao_extraction/`

## Tổng quan

Module này thực hiện trích xuất bộ ba **Subject / Action / Object (SAO)** từ câu chính sách kiểm soát truy cập ngôn ngữ tự nhiên (NLACP) để phục vụ hệ thống ABAC.

---

## Ranh giới Học thuật vs. Mở rộng Dự án

### [BASELINE: Alohaly et al. 2019]
Mã code bám sát bài báo:  
> Alohaly, M., Takabi, H., & Blanco, E. (2019). *Automated extraction of attributes from natural language ABAC policies*. Cybersecurity, Springer.

Trong paper gốc:
- **Subject** và **Object** là **input** của bộ phân loại CNN (phân loại quan hệ attribute-value), **không phải output trực tiếp**.
- Paper tập trung vào phân loại attribute-value pairs (ví dụ: role → "doctor", resource → "patient file").
- `AttributeConstraint` (relation, value) trong schema của dự án phản ánh khái niệm attribute-value pair trong paper.

### [EXTENDED] — Mở rộng của dự án này
Phần **không thuộc** paper gốc:
- Trích xuất triplet S/A/O tường minh dưới dạng `SAOTriplet` dataclass.
- Pipeline dependency parsing 8 bước (`SAOPipeline` + các `PipelineComponent`).
- `GazetteerValidator` với confidence scoring.
- `RuleBasedSAOExtractor` strategy.
- Toàn bộ `components/` package.

---

## Kiến trúc

```
sao_extraction/
├── core/
│   ├── interfaces.py          # Abstract: SAOExtractorStrategy, GazetteerProvider, PipelineComponent
│   ├── schemas.py             # Dataclass: SAOTriplet, PipelineResult, PipelineContext, AttributeConstraint
│   └── pipeline.py            # SAOPipeline orchestrator (from_config + component enable/disable)
├── components/
│   ├── preprocessing.py       # Contraction expansion + whitespace normalization
│   ├── clause_segmentation.py # Clause splitting via conj/cc at ROOT
│   ├── voice_polarity.py      # Voice (nsubjpass/auxpass) + Effect (Gazetteer) + Negation (neg dep)
│   ├── subject_extraction.py  # nsubj / agent>pobj + NP expansion (relcl excluded)
│   ├── action_extraction.py   # xcomp priority + prt phrasal verb
│   ├── object_extraction.py   # dobj/attr/oprd + prep>pobj + conj
│   ├── attribute_attachment.py# PP-modifier + relcl → AttributeConstraint
│   └── gazetteer_validation.py# Confidence scoring + SAOTriplet assembly
├── gazetteers/
│   ├── provider.py            # FileBasedGazetteer + CompositeGazetteer
│   └── data/
│       ├── effect_terms.json      # {"permit": [...], "deny": [...]}
│       ├── subject_roles.json     # {"roles": [...]}
│       └── resource_types.json    # {"resources": [...]}
├── config/
│   └── pipeline_config.yaml   # version, model_name, components bật/tắt
├── extractors/
│   └── rule_based_extractor.py # RuleBasedSAOExtractor implements SAOExtractorStrategy
└── tests/
    ├── fixtures/policy_samples.json   # 25 câu mẫu (5 nhóm × 5 câu)
    ├── test_clause_segmentation.py
    ├── test_subject_extraction.py
    ├── test_action_extraction.py
    ├── test_object_extraction.py
    └── test_gazetteer_validation.py
```

---

## Thứ tự Pipeline (Version 1)

```
Input text
   │
   ▼
1. PreprocessingComponent      → normalize text (contractions, whitespace)
   │
   ▼
2. ClauseSegmenter             → split into independent clause spans
   │
   ▼
3. VoicePolarityDetector       → voice + effect + negation per clause
   │
   ▼
4. SubjectExtractor            → nsubj / agent>pobj + NP expansion
   │
   ▼
5. ActionExtractor             → xcomp priority + phrasal verb prt
   │
   ▼
6. ObjectExtractor             → dobj/attr/oprd + prep>pobj + conj
   │
   ▼
7. AttributeConstraintExtractor→ PP-modifier + relcl → AttributeConstraint
   │
   ▼
8. GazetteerValidator          → confidence scoring + SAOTriplet assembly
   │
   ▼
PipelineResult (triplets, version, warnings)
```

---

## Cách sử dụng

```python
from backend.sao_extraction.extractors.rule_based_extractor import RuleBasedSAOExtractor

extractor = RuleBasedSAOExtractor()
result = extractor.extract("The attending physician can review patient files.")

for triplet in result.triplets:
    print(triplet.subject)    # "attending physician"
    print(triplet.action)     # "review"
    print(triplet.objects)    # ["patient files"]
    print(triplet.effect)     # "Permit"
    print(triplet.confidence) # e.g. 1.1 (capped to 1.0)
```

---

## Chạy Tests

```bash
# Unit tests từng component
python -m pytest backend/sao_extraction/tests/ -v

# Integration tests (đảm bảo main.py adapter không bị phá)
python -m pytest tests/ -v
```

---

## Giới hạn đã biết (Version 1)

- **Tiếng Việt**: spaCy không có model dependency parsing tiếng Việt. Các câu tiếng Việt sẽ cho kết quả kém hơn so với tiếng Anh. Tính năng tiếng Việt sẽ được phát triển trong phiên bản tương lai.
- **Câu phức tạp**: Pipeline xử lý tốt các câu đơn, bị động, phủ định, và coordination cơ bản. Các cấu trúc lồng phức tạp nhiều tầng có thể không hoàn toàn chính xác.
- **Coreference**: Pipeline không giải quyết coreference (ví dụ: "they" → "doctors"). Tính năng này thuộc phạm vi Version 3/4.

---

## Changelog & Roadmap

| Version | Status | Mô tả |
|---------|--------|-------|
| **v1.0** | ✅ Hiện tại | Rule-based / Pure-NLP (dependency parsing + Gazetteer) |
| v2.0 | 🔜 Dự kiến | Hybrid Rule + CNN — thêm CNN classifier cho attribute-value (bám sát Alohaly et al. 2019) |
| v3.0 | 🔜 Dự kiến | SRL layer — Semantic Role Labeling (AllenNLP / spaCy) |
| v4.0 | 🔜 Dự kiến | LLM few-shot — prompting LLM với few-shot examples từ `policy_samples.json` |

---

## Mở rộng kiến trúc

### Thêm strategy mới (ví dụ: Version 2 — CNN)
```python
# extractors/cnn_extractor.py
class HybridCNNExtractor(SAOExtractorStrategy):
    def extract(self, text: str) -> PipelineResult: ...
    @property
    def strategy_name(self) -> str: return "cnn"
```
Không cần sửa bất kỳ component nào hoặc `main.py`.

### Thay Gazetteer nguồn
```python
# Thay FileBasedGazetteer bằng DatabaseGazetteer
extractor = RuleBasedSAOExtractor(gazetteer=DatabaseGazetteer(connection_string="..."))
```

### Tắt component để ablation study
Trong `pipeline_config.yaml`:
```yaml
components:
  attribute_attachment: false  # Tắt để đo ảnh hưởng
```
