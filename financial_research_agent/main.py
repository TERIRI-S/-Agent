"""
智能金融研报 + 风控合规多 Agent 系统
==========================================
使用 LangGraph 构建，包含四个 Agent：
1. 数据采集 Agent
2. 事实提取 Agent
3. 合规审查 Agent
4. 交叉验证 Agent
最终输出带合规评分的研报。
"""

import json
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseLLM


# ------- 1. 状态定义 -------
class ResearchState(TypedDict):
    stock_symbol: str
    raw_data: str
    facts: list
    draft_report: str
    compliance_issues: list
    verified_facts: list
    final_report: str
    compliance_score: float


# ------- 2. 模拟 LLM（可替换为真实 API） -------
class MockLLM(BaseLLM):
    """Fake LLM 用于演示逻辑"""
    @property
    def _llm_type(self) -> str:
        return "mock"

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        if "数据采集" in prompt:
            return f"""以下是关于{self.stock_symbol}的最新资讯和公告：
1. 公告：{self.stock_symbol} 2025财年营收340亿元，同比增长15%，净利润42亿元。
2. 新闻：{self.stock_symbol}宣布与某新能源车企达成战略合作，共同开发固态电池。
3. 公告：公司拟以自有资金回购股份，金额不超过10亿元。
4. 社交媒体：传闻董事长可能减持，但公司尚未回应。
"""
        elif "提取以下文本中的关键事实" in prompt:
            return json.dumps([
                {"指标": "营收", "值": "340亿元", "时间": "2025财年"},
                {"指标": "净利润", "值": "42亿元", "时间": "2025财年"},
                {"指标": "合作事件", "值": "与新能源车企合作固态电池"},
                {"指标": "回购计划", "值": "不超过10亿元"},
                {"指标": "传闻", "值": "董事长可能减持"}
            ], ensure_ascii=False)
        elif "合规审查" in prompt:
            return json.dumps([
                {"问题": "传闻信息（董事长减持）未经证实，需标注'市场传闻，待核实'"},
                {"问题": "回购金额表述缺乏正式公告原文链接"}
            ], ensure_ascii=False)
        elif "交叉验证" in prompt:
            return json.dumps([
                {"指标": "营收", "原始值": "340亿元", "验证结果": "已核对：官方年报数据340.2亿，一致"},
                {"指标": "净利润", "原始值": "42亿元", "验证结果": "已核对：官方年报数据42.1亿，误差可忽略"},
                {"指标": "回购计划", "原始值": "不超过10亿元", "验证结果": "已核对：交易所公告已发布，一致"},
                {"指标": "传闻", "原始值": "董事长可能减持", "验证结果": "无法在权威渠道确认，标记为高风险"}
            ], ensure_ascii=False)
        elif "生成最终研报" in prompt:
            facts_str = "\n".join(str(f) for f in self.verified_facts)
            issues_str = "\n".join(i["问题"] for i in self.compliance_issues)
            return f"""
【{self.stock_symbol} 深度研报】
基本事实（已交叉验证）：
{facts_str}

合规审查结果：
{issues_str}

风险提示：部分传闻未经证实，投资者需关注后续公告。
免责声明：本报告由AI生成，仅供参考，不构成投资建议。
"""
        return "[]"

    @property
    def stock_symbol(self):
        return self._stock_symbol

    @stock_symbol.setter
    def stock_symbol(self, val):
        self._stock_symbol = val

    @property
    def verified_facts(self):
        return getattr(self, '_verified_facts', [])

    @verified_facts.setter
    def verified_facts(self, val):
        self._verified_facts = val

    @property
    def compliance_issues(self):
        return getattr(self, '_compliance_issues', [])

    @compliance_issues.setter
    def compliance_issues(self, val):
        self._compliance_issues = val


# ------- 3. Agent 节点定义 -------
class ResearchAgents:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def data_collector(self, state: ResearchState) -> ResearchState:
        self.llm.stock_symbol = state['stock_symbol']
        prompt = f"数据采集：请提供 {state['stock_symbol']} 的最新媒体报道和公告。"
        state['raw_data'] = self.llm(prompt)
        return state

    def fact_extractor(self, state: ResearchState) -> ResearchState:
        prompt = f"提取以下文本中的关键事实（JSON 格式）：\n{state['raw_data']}"
        facts_json = self.llm(prompt)
        state['facts'] = json.loads(facts_json)
        return state

    def compliance_reviewer(self, state: ResearchState) -> ResearchState:
        prompt = f"""请对以下事实进行合规审查，找出潜在违规点（如未证实传闻、误导性陈述）：
事实列表：{state['facts']}
输出 JSON 列表，每个元素包含 "问题" 字段。"""
        issues_json = self.llm(prompt)
        state['compliance_issues'] = json.loads(issues_json)
        return state

    def cross_validator(self, state: ResearchState) -> ResearchState:
        prompt = f"""请将以下事实与权威数据源比对验证，输出每条数据的验证结果：
事实：{state['facts']}
返回 JSON 列表，包含原始值、验证结果。"""
        verified_json = self.llm(prompt)
        self.llm.verified_facts = json.loads(verified_json)
        self.llm.compliance_issues = state['compliance_issues']
        state['verified_facts'] = self.llm.verified_facts
        return state

    def report_generator(self, state: ResearchState) -> ResearchState:
        self.llm.verified_facts = state['verified_facts']
        self.llm.compliance_issues = state['compliance_issues']
        prompt = "生成最终研报"
        report = self.llm(prompt)
        state['final_report'] = report
        state['compliance_score'] = max(0, 100 - len(state['compliance_issues']) * 20)
        return state


# ------- 4. 构建 LangGraph 工作流 -------
def build_graph():
    llm = MockLLM()
    agents = ResearchAgents(llm)

    workflow = StateGraph(ResearchState)

    workflow.add_node("data_collector", agents.data_collector)
    workflow.add_node("fact_extractor", agents.fact_extractor)
    workflow.add_node("compliance_reviewer", agents.compliance_reviewer)
    workflow.add_node("cross_validator", agents.cross_validator)
    workflow.add_node("report_generator", agents.report_generator)

    workflow.set_entry_point("data_collector")
    workflow.add_edge("data_collector", "fact_extractor")
    workflow.add_edge("fact_extractor", "compliance_reviewer")
    workflow.add_edge("compliance_reviewer", "cross_validator")
    workflow.add_edge("cross_validator", "report_generator")
    workflow.add_edge("report_generator", END)

    return workflow.compile()


# ------- 5. 运行演示 -------
if __name__ == "__main__":
    app = build_graph()
    initial_state = {
        "stock_symbol": "宁德时代",
        "raw_data": "",
        "facts": [],
        "draft_report": "",
        "compliance_issues": [],
        "verified_facts": [],
        "final_report": "",
        "compliance_score": 0.0,
    }
    print("===== 多 Agent 金融研报系统启动 =====")
    result = app.invoke(initial_state)

    print("\n----- 最终研报 -----")
    print(result["final_report"])
    print(f"\n合规评分: {result['compliance_score']}/100")