"""Build a held-out evaluation set, disjoint from ``bge_m3_lite/calibration.txt``.

Writes 40 diverse texts to ``tests/fixtures/heldout_texts.json`` and the fp32
FUSED-model reference dense/sparse/colbert outputs to
``tests/fixtures/heldout_ref.npz`` (+ ``heldout_ref.json`` for the sentences
and sparse dicts), the same way ``tools/make_embedding_fixtures.py`` does for
the FlagEmbedding fixtures -- except the reference here is our own fp32 fused
graph (bit-exact with FlagEmbedding per docs/verification.md), so no extra
dependency (torch/FlagEmbedding) is needed to regenerate it.

Run:  uv run python tools/make_heldout.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bge_m3_lite.embedder import BGEM3Embedder

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

_LONG_EN = (
    "The history of computing is often told as a story of exponential progress, "
    "but it is really a story of accumulated abstraction. Early machines were "
    "programmed by rewiring physical plugboards, one instruction encoded as a "
    "literal connection between two terminals; a single mistake meant hours of "
    "retracing wires rather than editing a line of text. The invention of stored "
    "program architectures collapsed that distinction between instructions and "
    "data, letting a machine treat its own program as just another array of "
    "numbers it could inspect, copy, and even modify while running. Compilers "
    "then added a further layer, translating human-readable statements into the "
    "machine code that earlier engineers had to write by hand, and with every "
    "such layer the cost of expressing an idea dropped while the cost of running "
    "it barely changed. Operating systems multiplexed a single processor across "
    "many programs, databases gave those programs a shared and durable notion of "
    "state, and networks let programs on different machines exchange that state "
    "as though distance did not matter. None of this made the underlying "
    "hardware simpler; if anything, a modern chip is more intricate than any "
    "system a nineteen-fifties engineer could have described end to end. What "
    "changed is that almost nobody needs to hold the whole stack in their head "
    "at once anymore. A web developer can build a working application without "
    "knowing how the processor decodes an instruction, and a hardware engineer "
    "can design a new chip without knowing what applications will eventually run "
    "on it. This separation of concerns is what let the field scale: thousands "
    "of people can improve their own layer independently, trusting that the "
    "layers below and above will keep working as expected. The same pattern "
    "shows up again with machine learning, where a researcher can now fine-tune "
    "a large pretrained model without understanding the linear algebra kernels "
    "that make a single forward pass fast, and a systems engineer can optimise "
    "those kernels without understanding what the model is being used for. "
    "Abstraction always has a cost, usually paid in efficiency lost at the "
    "boundary between layers, and periodically someone rediscovers that cost is "
    "worth paying to look underneath one more layer than everybody else "
    "bothered to, which is often exactly how the next big jump in performance "
    "gets found. It is a slow, uneven kind of progress, punctuated by moments "
    "where an old constraint quietly disappears and nobody notices for years."
)

_LONG_ZH = (
    "在中国古代，造纸术、印刷术、火药和指南针被合称为四大发明，它们不仅深刻改变了"
    "中国社会的发展轨迹，也通过丝绸之路和海上贸易路线传播到欧亚大陆的其他地区，"
    "间接推动了欧洲文艺复兴和近代科学的兴起。造纸术最早出现于西汉时期，东汉时期"
    "蔡伦改进了造纸工艺，使纸张的成本大幅降低，逐渐取代竹简和丝帛成为主要的书写"
    "材料。印刷术则经历了从雕版印刷到活字印刷的演变，北宋时期毕昇发明的胶泥活字"
    "印刷术，比欧洲古腾堡的金属活字印刷早了大约四百年。火药最初被方士用于炼丹，"
    "唐代之后逐渐被应用于军事领域，制造出火箭、火铳等早期火器。指南针的前身是"
    "战国时期的司南，宋代以后被广泛用于航海，为大航海时代的到来提供了关键的技术"
    "支持。这些发明的传播路径十分复杂，往往经过阿拉伯商人和蒙古帝国的中介，辗转"
    "数百年才最终抵达欧洲，而在传播过程中技术本身也在不断被改良和重新组合。"
)

HELDOUT_TEXTS: list[str] = [
    # -- one sentence per language (not present in calibration.txt) --------
    "The annual shareholder meeting will be held virtually this year due to renovations at the main office.",  # en
    "上海自贸区近年来吸引了大量外资企业入驻，成为长三角地区经济发展的重要引擎。",  # zh
    "新しいスマートフォンのバッテリー持続時間は、従来モデルと比べて約二割向上しています。",  # ja
    "이번 겨울철 전력 수요 급증에 대비해 정부는 예비 전력 확보 방안을 마련하고 있다.",  # ko
    "Der Zug nach München hat wegen eines technischen Defekts eine Verspätung von etwa zwanzig Minuten.",  # de
    "La bibliothèque municipale prolonge ses horaires d'ouverture pendant la période des examens universitaires.",  # fr
    "El nuevo puente peatonal conectará el centro histórico con el parque situado al otro lado del río.",  # es
    "Учёные из нескольких стран объединили усилия для изучения таяния арктических льдов.",  # ru
    "افتتحت الحكومة أمس مطارًا جديدًا يهدف إلى تخفيف الازدحام عن المطار الرئيسي في العاصمة.",  # ar
    "स्थानीय प्रशासन ने बाढ़ प्रभावित क्षेत्रों में राहत शिविर स्थापित करने का निर्णय लिया है।",  # hi
    "ทางการเมืองประกาศมาตรการช่วยเหลือเกษตรกรที่ประสบภัยแล้งในภาคตะวันออกเฉียงเหนือ",  # th
    "Chính quyền địa phương vừa công bố kế hoạch mở rộng tuyến đường sắt đô thị trong năm năm tới.",  # vi
    # -- code snippets -------------------------------------------------------
    "class LRUCache:\n    def __init__(self, capacity):\n        self.cache = {}\n        self.capacity = capacity\n    def get(self, key):\n        return self.cache.get(key, -1)",  # python
    "async function fetchWithRetry(url, retries = 3) {\n  try {\n    return await fetch(url);\n  } catch (err) {\n    if (retries === 0) throw err;\n    return fetchWithRetry(url, retries - 1);\n  }\n}",  # js
    # -- structured text -------------------------------------------------------
    "| Model | Params | Accuracy | Latency (ms) |\n|---|---|---|---|\n| tiny | 5M | 0.71 | 3 |\n| base | 110M | 0.89 | 18 |\n| large | 340M | 0.93 | 52 |",  # markdown table
    '{"user_id": 8842, "roles": ["admin", "editor"], "active": true, "quota": {"storage_gb": 50, "used_gb": 12.4}, "tags": null}',  # json blob
    _LONG_EN,  # ~600-token English paragraph
    _LONG_ZH,  # long Chinese paragraph
    # -- numbers, dates, misc formats -----------------------------------------
    "Invoice #2024-0091 dated 2024-11-03: subtotal $1,284.50, tax 8.25% ($105.97), total $1,390.47, due 2024-12-03.",
    "The marathon record improved from 2:01:39 to 2:00:35 over three consecutive Berlin races (2018, 2022, 2023).",
    # -- emojis ----------------------------------------------------------------
    "packing for the trip ✈️🧳☀️ can't wait to see the beach again 🏖️🐚 see you all in two weeks! 👋😊",
    # -- empty-ish / degenerate --------------------------------------------
    "ok",
    "   \n\t  ",
    # -- questions -------------------------------------------------------------
    "Why does my flight keep getting delayed even though the weather looks fine on the radar?",
    "为什么我的电脑在连接外接显示器后风扇声音会突然变大？",
    # -- mixed language ----------------------------------------------------
    "The meeting agenda (議程) covers Q3 財報, roadmap 2025, and 予算 review — please read 事前に before Thursday.",
    # -- longer paragraphs in other languages -------------------------------
    "Die Digitalisierung der öffentlichen Verwaltung schreitet in vielen Bundesländern nur langsam voran, "
    "obwohl seit Jahren gesetzliche Fristen für die Bereitstellung von Online-Diensten bestehen. Kritiker "
    "bemängeln vor allem die uneinheitliche technische Infrastruktur zwischen Kommunen, die dazu führt, dass "
    "Bürgerinnen und Bürger je nach Wohnort völlig unterschiedliche Erfahrungen mit Behördengängen machen.",  # de long
    "Le débat sur la transition énergétique s'intensifie à mesure que les échéances climatiques se "
    "rapprochent, avec des positions parfois opposées entre les partisans d'un déploiement massif de "
    "l'éolien et ceux qui privilégient le nucléaire comme solution bas carbone la plus fiable à long terme.",  # fr long
    "Экономисты расходятся во мнениях относительно долгосрочных последствий текущей денежно-кредитной "
    "политики центрального банка, отмечая, что снижение ставки может как стимулировать инвестиции, так и "
    "спровоцировать новый виток инфляции в случае резкого роста потребительского спроса.",  # ru long
    "أعلنت وزارة الصحة عن حملة تطعيم واسعة النطاق تستهدف المناطق الريفية النائية، مشيرة إلى أن الهدف هو "
    "الوصول إلى نسبة تغطية تتجاوز تسعين بالمئة خلال الأشهر الستة المقبلة رغم التحديات اللوجستية الكبيرة.",  # ar long
    "स्वास्थ्य विशेषज्ञों का कहना है कि नियमित व्यायाम और संतुलित आहार न केवल शारीरिक स्वास्थ्य बल्कि मानसिक "
    "स्वास्थ्य पर भी सकारात्मक प्रभाव डालते हैं, और इसीलिए स्कूलों में शारीरिक शिक्षा को अनिवार्य किया जाना चाहिए।",  # hi long
    "นักวิเคราะห์เศรษฐกิจระบุว่าอัตราเงินเฟ้อที่เพิ่มสูงขึ้นในช่วงหลายเดือนที่ผ่านมาส่งผลกระทบโดยตรงต่อกำลังซื้อ"
    "ของประชาชน โดยเฉพาะกลุ่มผู้มีรายได้น้อยที่ต้องแบกรับภาระค่าครองชีพที่สูงขึ้นอย่างต่อเนื่อง",  # th long
    "Nhiều chuyên gia giáo dục cho rằng chương trình học hiện tại đặt quá nhiều áp lực thi cử lên học sinh "
    "trung học, dẫn đến tình trạng căng thẳng kéo dài và ảnh hưởng tiêu cực đến sức khỏe tinh thần của các em.",  # vi long
    "최근 몇 년간 국내 스타트업 생태계는 대규모 투자 유치보다는 수익성 중심의 성장 전략으로 방향을 전환하고 "
    "있으며, 이는 글로벌 금리 인상과 투자 심리 위축이라는 외부 환경 변화에 대응한 결과로 분석된다.",  # ko long
    "Los expertos en salud pública advierten que la resistencia a los antibióticos podría convertirse en "
    "una de las principales causas de mortalidad a nivel mundial si no se adoptan medidas urgentes para "
    "regular su uso tanto en la medicina humana como en la producción agrícola y ganadera.",  # es long
    # -- lists / bullets ------------------------------------------------------
    "Weekend checklist:\n- water the plants\n- return library books\n- call the plumber about the leak\n- prep slides for Monday's review",
    # -- code + prose mix -------------------------------------------------
    "The bug only reproduces when `batch_size > 1`: `assert out.shape[0] == batch_size` fails because the "
    "padding mask is never applied before the final `Reshape`, so trailing rows silently pick up garbage.",
    # -- social-media style ---------------------------------------------------
    "just spent three hours debugging a typo in a config file #mondaymood #programmerlife send help",
    # -- more numeric / date heavy ---------------------------------------
    "Q1 2025 revenue: $4.2M (+18% YoY); Q2: $4.6M (+9.5%); Q3 forecast: $5.1M, contingent on the EU launch closing by 03/15.",
    # -- extra short fragment --------------------------------------------
    "no.",
]

assert len(HELDOUT_TEXTS) == 40, len(HELDOUT_TEXTS)


def _load_calibration_lines() -> set[str]:
    from bge_m3_lite.quantize import load_calibration_texts

    return set(load_calibration_texts())


def main() -> None:
    calib = _load_calibration_lines()
    overlap = [t for t in HELDOUT_TEXTS if t.strip() and t in calib]
    if overlap:
        raise SystemExit(f"held-out texts overlap calibration.txt: {overlap!r}")

    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "heldout_texts.json").write_text(
        json.dumps(HELDOUT_TEXTS, ensure_ascii=False, indent=0) + "\n",
        encoding="utf-8",
    )

    embedder = BGEM3Embedder(quiet=True)  # fp32 fused, bit-exact with FlagEmbedding
    out = embedder.encode(
        HELDOUT_TEXTS,
        batch_size=4,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    embedder.close()

    dense = np.asarray(out["dense_vecs"], dtype=np.float32)
    arrays: dict[str, np.ndarray] = {"dense": dense}
    for i, v in enumerate(out["colbert_vecs"]):
        arrays[f"colbert_{i}"] = np.asarray(v, dtype=np.float32)
    np.savez_compressed(FIXTURES / "heldout_ref.npz", **arrays)  # pyright: ignore[reportArgumentType]

    lexical = [
        {str(k): float(v) for k, v in lw.items()} for lw in out["lexical_weights"]
    ]
    (FIXTURES / "heldout_ref.json").write_text(
        json.dumps(
            {"sentences": HELDOUT_TEXTS, "lexical_weights": lexical},
            ensure_ascii=False,
            indent=0,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(HELDOUT_TEXTS)} texts, dense {dense.shape}")


if __name__ == "__main__":
    main()
