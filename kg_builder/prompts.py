"""Prompt templates for the LLM-backed agents.

All extraction prompts ask for strict JSON so the agents can parse results
deterministically.
"""
from __future__ import annotations

ENTITY_TYPES = [
    "Person", "Company", "Organization", "Product", "Technology",
    "Concept", "Event", "Location", "Date", "Metric", "Other",
]

EXTRACTION_SYSTEM = (
    "你是一名通用的知识图谱信息抽取专家，适用于新闻、研报、百科、书籍、对话、"
    "规则/心得等各类文本。只输出 JSON，不要输出任何解释性文字。"
    "从文本中抽取实体、实体间关系以及事件。"
)

# 两阶段抽取：先抽实体（紧凑），再把实体名回灌用于专抽关系，
# 避免模型在单一 JSON 中把输出预算耗尽在实体上而导致关系缺失。
EXTRACTION_ENTITIES_PROMPT = """请从下面的文本中抽取实体，并严格按如下 JSON 结构返回（不要输出任何解释）：

{
  "entities": [
    {"name": "实体名", "type": "类型", "aliases": ["别名1", "别名2"], "description": "一句话简述（不超过 12 字）"}
  ]
}

实体类型建议从以下集合选取（也可根据文本自行归纳）：{types}。
其中 Person=人物, Company=公司, Organization=组织/机构, Product=产品, Technology=技术/方法,
Concept=概念/观点/原则, Event=事件, Location=地点, Date=时间, Metric=指标/数值, Other=其他。

抽取原则（通用，不限领域）：
- 既抽取具体对象（人物、组织、地点、产品、技术），也抽取抽象对象（概念、方法、原则、指标、事件）。
- 对规则/心得/说明类文本，请提炼其中“统领性”的核心概念（type=Concept），不要逐条罗列。
- 同一对象的不同表述归并为一个实体，其余放入 aliases。
- 严禁把“第N条 / 第N步 / N 条法则 / N 个步骤”这类编号本身当作实体，也不要为每条目单独建实体；应只抽取少数真正重要的概念。
- 只返回最重要的核心实体（单阶段模式不超过 8 个，两阶段模式不超过 20 个），描述务必简短。

文本：
{text}
"""

EXTRACTION_RELATIONS_PROMPT = """已知候选实体列表：
{entities}

请从下面的文本中抽取上述实体之间的关系，并严格按如下 JSON 结构返回（不要输出任何解释）：

{
  "relations": [
    {"source": "头实体名", "target": "尾实体名", "type": "关系类型(大写英文, 如 PART_OF / WORKS_FOR / USES / RELATED_TO)"}
  ]
}

抽取原则（通用，不限领域）：
- 关系须基于文本事实；source / target 必须是上面候选实体列表中真实存在的名称（不得凭空造实体，也不得把“第N条”等编号当作实体）。
- 关系类型用简洁大写英文表达语义，跨领域通用，例如：
  · 人物/组织：WORKS_FOR(任职于) / FOUNDED(创立) / LEADS(领导) / MEMBER_OF(属于) / COLLABORATES_WITH(合作)
  · 组成/归属：PART_OF(部分属于) / BELONGS_TO(归属) / CONTAINS(包含) / LOCATED_IN(位于)
  · 因果/影响：CAUSES(导致) / INFLUENCES(影响) / DEPENDS_ON(依赖) / PREVENTS(防止)
  · 行为/使用：USES(使用) / PRODUCES(生产) / PROVIDES(提供) / MENTIONS(提及)
  · 概念/主题：RELATED_TO(相关) / EXPRESSES(表达) / COMPARES(对比) / SUPPORTS(支持) / CONTRADICTS(相悖)
- 对规则/心得/说明类文本，也要抽取概念之间的逻辑关联（如“方法 A 有助于 原则 B”“概念 X 对应 概念 Y”）。
- 若文本是清单/条目式内容（如“N 条法则”“N 个步骤”），不要为每条目单独建立关系，只抽取贯穿全文或概念级的有意义关联。
- 关系是图谱的骨架，请尽量抽取有意义（含概念间逻辑）的关联，最多 20 条；若文本确实没有任何实体间关系，返回空数组。

文本：
{text}
"""

EXTRACTION_PROMPT = """请从下面的文本中抽取信息，并严格按如下 JSON 结构返回：

{
  "entities": [
    {"name": "实体名", "type": "类型", "aliases": ["别名1", "别名2"], "description": "一句话简述（不超过 12 字）"}
  ],
  "relations": [
    {"source": "头实体名", "target": "尾实体名", "type": "关系类型(大写英文, 如 PART_OF / WORKS_FOR / USES / RELATED_TO)"}
  ],
  "events": [
    {"type": "事件类型", "subject": "主体", "object": "客体", "time": "时间", "summary": "一句话摘要"}
  ]
}

实体类型建议从以下集合选取（也可自行归纳）：{types}。

【输出要求：质量优先，宁可少而精】
- entities 不超过 8 个，覆盖人物 / 组织 / 地点 / 产品 / 技术 / 概念 / 事件等核心实体。规则/心得类文本只提炼 1~3 个统领性核心概念，严禁逐条罗列“第N条法则/第N步”等条目，也不要把编号本身当实体。
- relations 不超过 12 条，且 source / target 必须来自上面 entities 中的 name（不得凭空造实体）；关系是图谱骨架，请只抽取真正有意义的关联。
- events 不超过 6 个。
【重要】若文本是清单/条目式内容（如“N 条法则”“N 个步骤”），不要为每条目单独建立关系，也不要把“第N条”当作实体；只抽取贯穿全文或概念级别的有意义关联。
若文本较长，只保留最关键的信息，不要穷举。

文本：
\"\"\"
{text}
\"\"\"
"""

# 合并抽取（MERGE_UNDERSTANDING_EXTRACTION=true 时使用）：
# 把"语义理解(事件)"合并进"实体关系抽取"阶段。逻辑为两步（保证关系/事件可靠产出）：
#   第 1 步：抽实体（紧凑）；第 2 步：给定实体名，一次抽取 关系 + 事件。
# 相比原两段式（理解 1 次 + 抽取两段 = 3 次调用），每 chunk 降为 2 次调用。
MERGED_REL_EVENTS_PROMPT = """已知候选实体列表：
{entities}

请从上面的文本中抽取这些实体之间的关系，以及文本中的关键事件，并严格按如下 JSON 返回：

{
  "relations": [
    {"source": "头实体名", "target": "尾实体名", "type": "关系类型(大写英文, 如 PART_OF / WORKS_FOR / USES / CAUSES / FOUNDED / LEADS / COLLABORATES_WITH / RELATED_TO)"}
  ],
  "events": [
    {"type": "事件类型", "subject": "主体(实体名)", "object": "客体(实体名)", "time": "时间", "summary": "一句话摘要"}
  ]
}

抽取原则（通用）：
- 关系须基于文本事实；source / target 必须来自上面的候选实体列表（不要凭空造实体）。
- 关系类型用简洁大写英文表达语义，跨领域通用，例如：
  · 人物/组织：WORKS_FOR(任职于) / FOUNDED(创立) / LEADS(领导) / MEMBER_OF(属于) / COLLABORATES_WITH(合作)
  · 组成/归属：PART_OF(部分属于) / BELONGS_TO(归属) / CONTAINS(包含) / LOCATED_IN(位于)
  · 因果/影响：CAUSES(导致) / INFLUENCES(影响) / DEPENDS_ON(依赖) / PREVENTS(防止)
  · 行为/使用：USES(使用) / PRODUCES(生产) / PROVIDES(提供) / MENTIONS(提及)
  · 概念/主题：RELATED_TO(相关) / EXPRESSES(表达) / COMPARES(对比) / SUPPORTS(支持) / CONTRADICTS(相悖)
- 事件描述文本中发生的关键事件：谁(subject)在何时(time)对谁(object)做了什么，summary 一句话概括。subject / object 必须来自候选实体。
- 若文本是清单/条目式内容（如“N 条法则”“N 个步骤”），不要为每条目单独建关系，只抽取概念级关联。
- 关系是图谱骨架，请尽量抽取有意义（含概念间逻辑）的关联，最多 20 条；事件不超过 6 个；若文本确实没有任何关系/事件，返回空数组。

文本：
{text}
"""

# 抽取后校正：把初步抽取结果连同原文再次交给 LLM，修正实体名/关系端点/类型等，
# 达到"修正结果"的目的。用于 EXTRACTION_CORRECTION=true。
EXTRACTION_CORRECTION_PROMPT = """你是一名知识图谱抽取结果的“校正”审核专家。下面给出【原文】与模型给出的【初步抽取结果 JSON】。
请严格对照原文对抽取结果做校正，并仍按相同 JSON 结构返回（不要输出任何解释）：

{
  "entities": [{"name": "实体名", "type": "类型", "aliases": ["别名"], "description": "简述"}],
  "relations": [{"source": "头实体名", "target": "尾实体名", "type": "关系类型(大写英文)"}],
  "events": [{"type": "事件类型", "subject": "主体", "object": "客体", "time": "时间", "summary": "摘要"}]
}

校正规则（通用，对照原文）：
1. 实体名必须与原文用词一致：修正错别字、截断、多余标点、中英文混排错误（如“黄仁勋勋”→“黄仁勋”、“RTX30”→“RTX 30”）。
2. 同一对象的不同写法必须合并为一个实体（保留最规范名称为 name，其余写入 aliases）；不得出现重复实体。
3. 关系的 source / target 必须严格等于上方某个 entity 的 name（原样复制）；若关系引用的名称与实体名有出入（少字/多字/别名），请改为对应实体名；若完全无法对应到任何实体，则删除该关系。
4. 关系类型须为简洁大写英文；若明显错误或含糊（如“相关”“关系”），请修正为更准确的关系类型（参考：PART_OF / WORKS_FOR / CAUSES / USES / LEADS / FOUNDED / COLLABORATES_WITH / RELATED_TO 等）。
5. 事件 event 的 subject / object 也必须取自 entity 的 name；summary 须忠于原文，修正明显错误。
6. 只输出校正后的 JSON；不要新增原文中不存在的实体 / 关系 / 事件；不要改变原文事实。

【原文】
\"\"\"
{text}
\"\"\"

【初步抽取结果】
{data}
"""

UNDERSTANDING_SYSTEM = (
    "你是一名语义理解助手。只输出 JSON，不要输出任何解释性文字。"
    "理解文本语义，抽取其中表达的关键事件（谁在何时对谁做了什么）。"
)

UNDERSTANDING_PROMPT = """请理解下面文本并抽取关键事件，严格按如下 JSON 结构返回：

{
  "events": [
    {"type": "事件类型", "subject": "主体(实体名)", "object": "客体(实体名)", "time": "时间(如有)", "summary": "一句话摘要"}
  ]
}

文本：
\"\"\"
{text}
\"\"\"
"""

QA_ENTITY_LINKING_SYSTEM = (
    "你是一名知识图谱的查询理解助手（通用，不限领域）。"
    "请从用户问题中识别可用于图谱检索的关键实体，并判断其查询意图。"
)

QA_ENTITY_LINKING_PROMPT = """请分析下面的用户问题，提取其中可用于在知识图谱中检索的实体，并判断查询意图。

要求：
1. entities：问题中明确提到、可能作为图谱节点的实体名称。请使用标准名（例如用"苹果公司"而不要用"苹果"）；
   如能判断类型，请填写 type，可选：Person / Company / Organization / Product / Technology / Concept / Event / Location / Metric / Other；无法判断则留空字符串。
2. intent：一句话描述用户真正想问什么，例如"竞争优势""形成原因""对比分析""操作方法"。
3. 只输出 JSON，不要任何解释。

问题：
{question}

输出格式：
{
  "entities": [{"name": "实体名", "type": "类型"}],
  "intent": "查询意图"
}
"""

QA_SYSTEM = (
    "你是一名基于知识图谱与检索内容的知识问答助手（通用，不限领域）。"
    "请结合【知识图谱结构化关系】与【检索片段】回答问题。"
    "凡引用了图谱关系或检索片段中的事实，请在句末用括号标注出处，"
    "例如（来源：某报告.pdf-12）或（图谱：实体A-关系-实体B）。"
    "若信息不足请如实说明，不要编造。"
)

QA_PROMPT = """【用户问题】
{question}

（查询意图：{intent}）

【知识图谱结构化关系】（JSON 风格；● 为问题实体，○ 为关联实体；每条关系附带来源 sources）
{graph_context}

【检索片段】（已按与图谱实体的相关性 + 来源权重重排，含来源 meta）
{context}

回答要求：
1. 优先依据【知识图谱结构化关系】中的事实作答，尤其是实体间的直接关系；
2. 用【检索片段】补充细节与原文，但不得与图谱事实冲突；
3. 关键事实必须标注出处（图谱关系来源或检索片段来源文件/块）；
4. 若信息不足请如实说明，不要编造。
"""
