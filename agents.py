"""
DeFiScope — Multi-Agent Module
BaseAgent + ChainAgent, SentimentAgent, RiskProfileAgent, OrchestratorAgent

Modified: Gemini API → Local Ollama LLM (2026-03-15)
"""

import json
import time
import requests
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

try:
    from json import JSONDecodeError
except ImportError:
    JSONDecodeError = ValueError


# ──────────────────────────────────────────────
# Ollama Configuration — 换模型只改这里
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-2.5-flash-lite"


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────

@dataclass
class AgentOutput:
    agent_name: str
    output_json: dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        return {"agent_name": self.agent_name, "output": self.output_json, "timestamp": self.timestamp}


@dataclass
class GoalTree:
    root_goal: str
    sub_goals: list

    def to_dict(self):
        return {"root_goal": self.root_goal, "sub_goals": self.sub_goals}


@dataclass
class Recommendation:
    goal_tree: GoalTree
    chain_output: AgentOutput
    sentiment_output: AgentOutput
    risk_output: AgentOutput
    final_text: str
    sha256_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        return {
            "goal_tree": self.goal_tree.to_dict(),
            "chain_output": self.chain_output.to_dict(),
            "sentiment_output": self.sentiment_output.to_dict(),
            "risk_output": self.risk_output.to_dict(),
            "final_text": self.final_text,
            "sha256_hash": self.sha256_hash,
            "timestamp": self.timestamp,
        }


# ──────────────────────────────────────────────
# Data Snapshot Loader
# ──────────────────────────────────────────────

class DataSnapshot:
    def __init__(self, filepath: str = "mock_data.json"):
        self.filepath = filepath
        self.load_local()

    def load_local(self):
        with open(self.filepath, "r") as f:
            raw = json.load(f)
        self.protocols = raw["protocols"]
        self.news_articles = raw["news_articles"]

    def fetch_live_data(self):
        try:
            # Provide a standard User-Agent, some APIs block default Python requests
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get("https://api.llama.fi/protocols", headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            live_protocols = []
            for p in data[:50]:
                live_protocols.append({
                    "name": p.get("name", "Unknown"),
                    "chain": p.get("chain", "Multi"),
                    "category": p.get("category", "DeFi"),
                    "tvl": p.get("tvl", 0),
                    "tvl_change_24h": p.get("change_1d", 0) or 0,
                    "audit_status": "audited" if p.get("audits") and p.get("audits") != "0" else "unaudited",
                    "smart_contract_risk_score": 25 if p.get("audits") else 75,
                    "governance_score": 75 
                })
            
            self.protocols = live_protocols
            return True, "Success"
        except Exception as e:
            error_msg = str(e)
            print(f"Failed to fetch live data: {error_msg}")
            return False, error_msg

    def get_all_protocols(self):
        return self.protocols

    def get_articles_for_protocol(self, name: str):
        return [a for a in self.news_articles if a["protocol_name"].lower() == name.lower()]

    def get_protocol(self, name: str):
        for p in self.protocols:
            if p["name"].lower() == name.lower():
                return p
        return None


# ──────────────────────────────────────────────
# Base Agent  ★ Gemini → Ollama
# ──────────────────────────────────────────────

class BaseAgent:
    def __init__(self, agent_name: str, system_prompt: str, api_key: str = "", llm_provider: str = "ollama", gemini_model: str = GEMINI_MODEL):
        """
        api_key is used only when llm_provider="gemini".
        Local Ollama remains the default so the app still works offline.
        """
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.api_key = api_key
        self.llm_provider = llm_provider
        self.gemini_model = gemini_model

    def call_llm(self, prompt: str, require_json: bool = False) -> str:
        if self.llm_provider == "gemini":
            return self.call_gemini(prompt, require_json=require_json)
        return self.call_ollama(prompt, require_json=require_json)

    def call_ollama(self, prompt: str, require_json: bool = False) -> str:
        """Call local Ollama REST API and return text response."""
        full_prompt = f"{self.system_prompt}\n\n{prompt}"

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 4096,
            }
        }
        
        # Enforce JSON formatting if supported and requested
        if require_json:
            payload["format"] = "json"

        for attempt in range(3):
            try:
                print(f"[{self.agent_name}] Calling Ollama ({OLLAMA_MODEL})... (attempt {attempt + 1})")
                start = time.time()

                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json=payload,
                    timeout=180,
                )
                response.raise_for_status()

                result = response.json()
                elapsed = time.time() - start
                print(f"[{self.agent_name}] Done in {elapsed:.1f}s")
                return result.get("response", "")

            except requests.exceptions.ConnectionError:
                if attempt < 2:
                    print(f"[{self.agent_name}] Ollama not reachable, retrying in 3s...")
                    time.sleep(3)
                else:
                    raise RuntimeError(
                        f"[{self.agent_name}] Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                        "Make sure Ollama is running (open the Ollama app or run 'ollama serve')."
                    )
            except Exception as e:
                if attempt < 2:
                    print(f"[{self.agent_name}] Error: {e}, retrying...")
                    time.sleep(2)
                else:
                    raise RuntimeError(f"[{self.agent_name}] LLM call failed after 3 attempts: {e}")

    def call_gemini(self, prompt: str, require_json: bool = False) -> str:
        """Call Gemini generateContent REST API and return text response."""
        if not self.api_key:
            raise RuntimeError("Gemini API key is required when using Gemini.")

        payload = {
            "systemInstruction": {
                "parts": [{"text": self.system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 4096,
            },
        }

        if require_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        for attempt in range(3):
            try:
                print(f"[{self.agent_name}] Calling Gemini ({self.gemini_model})... (attempt {attempt + 1})")
                start = time.time()
                response = requests.post(
                    f"{GEMINI_BASE_URL}/models/{self.gemini_model}:generateContent",
                    headers=headers,
                    json=payload,
                    timeout=180,
                )
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait_hint = f" Wait about {retry_after} seconds before trying again." if retry_after else " Wait a minute before trying again."
                    raise RuntimeError(
                        f"[{self.agent_name}] Gemini rate/resource limit hit for {self.gemini_model}.{wait_hint} "
                        "One DeFiScope recommendation uses multiple Gemini calls, so visible dashboard usage can still look low."
                    )
                response.raise_for_status()
                result = response.json()
                elapsed = time.time() - start
                print(f"[{self.agent_name}] Done in {elapsed:.1f}s")

                parts = result.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text_parts = [part.get("text", "") for part in parts if part.get("text")]
                return "\n".join(text_parts).strip()

            except Exception as e:
                if "Gemini rate/resource limit hit" in str(e):
                    raise
                if attempt < 2:
                    print(f"[{self.agent_name}] Gemini error: {e}, retrying...")
                    time.sleep(2)
                else:
                    raise RuntimeError(f"[{self.agent_name}] Gemini call failed after 3 attempts: {e}")

    def parse_json(self, raw: str) -> dict:
        """Extract JSON from LLM response (handles markdown fences and <think> tags)."""
        import re
        text = raw.strip()
        
        # Remove <think>...</think> blocks common in local models
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```") and not l.strip().startswith("json")]
            text = "\n".join(lines).strip()
            
        try:
            return json.loads(text)
        except JSONDecodeError:
            # Try to find JSON block within the text via regex
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except JSONDecodeError:
                    pass
            repaired = self.repair_json_response(text)
            if repaired:
                return repaired
            raise ValueError(f"[{self.agent_name}] Could not parse JSON from response\nRaw Response: {raw[:200]}...")

    def repair_json_response(self, text: str) -> dict | None:
        """Best-effort repair for common LLM JSON issues such as truncated arrays."""
        start = text.find("{")
        if start == -1:
            return None

        candidate = text[start:].strip()
        if not candidate:
            return None

        for suffix in ("", "\n]}", "\n}", "\n]}"):
            try:
                return json.loads(candidate + suffix)
            except JSONDecodeError:
                continue
        return None

    def run(self, input_data: dict) -> AgentOutput:
        raise NotImplementedError


# ──────────────────────────────────────────────
# ChainAgent — On-Chain Protocol Risk Analysis
# ──────────────────────────────────────────────

CHAIN_AGENT_PROMPT = """You are ChainAgent, a DeFi on-chain analytics specialist.
Your job is to analyze DeFi protocol metrics and produce a risk assessment.

Given protocol data, you must:
1. Evaluate each protocol's risk based on TVL, TVL changes, audit status, smart contract risk, and governance score.
2. Flag any protocols with >10% TVL change in 24h as anomalies.
3. Rank protocols from lowest risk to highest risk.

Respond ONLY with a valid JSON object in this format:
{
  "protocol_assessments": [
    {
      "name": "ProtocolName",
      "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
      "composite_score": 20,
      "is_anomaly": false,
      "key_factors": ["factor1", "factor2"]
    }
  ],
  "market_summary": "Brief 2-sentence summary of the overall DeFi market risk landscape."
}"""


class ChainAgent(BaseAgent):
    def __init__(self, api_key: str, data_snapshot: DataSnapshot, llm_provider: str = "ollama", gemini_model: str = GEMINI_MODEL):
        super().__init__("ChainAgent", CHAIN_AGENT_PROMPT, api_key, llm_provider, gemini_model)
        self.data_snapshot = data_snapshot

    def run(self, input_data: dict = None) -> AgentOutput:
        protocols = sorted(
            self.data_snapshot.get_all_protocols(),
            key=lambda item: item.get("tvl", 0) or 0,
            reverse=True,
        )[:12]
        prompt = f"Analyze the following DeFi protocol data and produce a risk assessment:\n\n{json.dumps(protocols, indent=2)}"
        raw = self.call_llm(prompt, require_json=True)
        result = self.parse_json(raw)
        return AgentOutput(agent_name="ChainAgent", output_json=result)


# ──────────────────────────────────────────────
# SentimentAgent — Market Sentiment Analysis
# ──────────────────────────────────────────────

SENTIMENT_AGENT_PROMPT = """You are SentimentAgent, a crypto market sentiment analyst.
Your job is to classify sentiment from news articles about DeFi protocols.

Given news articles, you must:
1. Aggregate sentiment for each protocol (VERY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, VERY_POSITIVE).
2. Assign a sentiment score from -100 to +100 for each protocol.
3. Identify key sentiment signals (regulatory concerns, security incidents, growth milestones).

Respond ONLY with a valid JSON object in this format:
{
  "protocol_sentiments": [
    {
      "protocol_name": "ProtocolName",
      "sentiment": "POSITIVE",
      "sentiment_score": 65,
      "key_signals": ["signal1", "signal2"]
    }
  ],
  "overall_market_mood": "Brief 1-sentence summary of the overall market sentiment."
}"""


class SentimentAgent(BaseAgent):
    def __init__(self, api_key: str, data_snapshot: DataSnapshot, llm_provider: str = "ollama", gemini_model: str = GEMINI_MODEL):
        super().__init__("SentimentAgent", SENTIMENT_AGENT_PROMPT, api_key, llm_provider, gemini_model)
        self.data_snapshot = data_snapshot

    def run(self, input_data: dict = None) -> AgentOutput:
        articles = self.data_snapshot.news_articles
        # If no articles, still provide an empty array so LLM returns valid layout
        val = articles if articles else []
        prompt = f"Analyze the sentiment of the following crypto news articles:\n\n{json.dumps(val, indent=2)}"
        raw = self.call_llm(prompt, require_json=True)
        result = self.parse_json(raw)
        return AgentOutput(agent_name="SentimentAgent", output_json=result)


# ──────────────────────────────────────────────
# RiskProfileAgent — User Risk Matching
# ──────────────────────────────────────────────

RISK_PROFILE_PROMPT = """You are RiskProfileAgent, a personalized portfolio allocation specialist.
Your job is to match DeFi protocols to a user's risk profile and generate a portfolio allocation.

Given:
- A user's risk profile (category, score, horizon, experience, max drawdown)
- Protocol risk assessments from ChainAgent
- Sentiment analysis from SentimentAgent

You must:
1. Filter protocols compatible with the user's risk tolerance.
2. Rank suitable protocols considering both on-chain risk and sentiment.
3. Generate a recommended allocation with percentage weights (must sum to 100%).

Respond ONLY with a valid JSON object in this format:
{
  "recommended_allocation": [
    {
      "protocol": "ProtocolName",
      "weight_pct": 30,
      "rationale": "Brief reason for this allocation"
    }
  ],
  "excluded_protocols": [
    {
      "protocol": "ProtocolName",
      "reason": "Why excluded"
    }
  ],
  "portfolio_risk_summary": "Brief 2-sentence summary of the overall portfolio risk."
}"""


class RiskProfileAgent(BaseAgent):
    def __init__(self, api_key: str, llm_provider: str = "ollama", gemini_model: str = GEMINI_MODEL):
        super().__init__("RiskProfileAgent", RISK_PROFILE_PROMPT, api_key, llm_provider, gemini_model)

    def run(self, input_data: dict) -> AgentOutput:
        prompt = (
            f"User Risk Profile:\n{json.dumps(input_data['user_profile'], indent=2)}\n\n"
            f"ChainAgent Protocol Assessments:\n{json.dumps(input_data['chain_output'], indent=2)}\n\n"
            f"SentimentAgent Analysis:\n{json.dumps(input_data['sentiment_output'], indent=2)}\n\n"
            f"Generate a personalized portfolio allocation for this user."
        )
        raw = self.call_llm(prompt, require_json=True)
        result = self.parse_json(raw)
        return AgentOutput(agent_name="RiskProfileAgent", output_json=result)


# ──────────────────────────────────────────────
# OrchestratorAgent — Goal-Oriented Coordination
# ──────────────────────────────────────────────

ORCHESTRATOR_PROMPT = """You are OrchestratorAgent, the central coordinator of DeFiScope.
Your job is to decompose a user's investment query into a goal hierarchy following goal-oriented methodology,
then synthesize outputs from ChainAgent, SentimentAgent, and RiskProfileAgent into a unified recommendation.

When asked to DECOMPOSE a query, respond ONLY with a valid JSON goal tree:
{
  "root_goal": "The user's top-level investment objective",
  "sub_goals": [
    {
      "id": "G1",
      "description": "Sub-goal description",
      "assigned_agent": "ChainAgent|SentimentAgent|RiskProfileAgent",
      "sub_goals": []
    }
  ]
}

When asked to SYNTHESIZE, produce a comprehensive Markdown recommendation including:
1. Executive Summary (2-3 sentences)
2. Goal Achievement Status (which goals were met)
3. Recommended Allocation (table with protocol, weight, rationale)
4. Key Risks to Monitor
5. Disclaimer: "This is an analytical assessment for educational purposes only. Not financial advice."
"""


class OrchestratorAgent(BaseAgent):
    def __init__(self, api_key: str, chain_agent: ChainAgent, sentiment_agent: SentimentAgent, risk_profile_agent: RiskProfileAgent, llm_provider: str = "ollama", gemini_model: str = GEMINI_MODEL):
        super().__init__("OrchestratorAgent", ORCHESTRATOR_PROMPT, api_key, llm_provider, gemini_model)
        self.chain_agent = chain_agent
        self.sentiment_agent = sentiment_agent
        self.risk_profile_agent = risk_profile_agent

    def decompose_goals(self, query: str) -> GoalTree:
        prompt = (
            f"DECOMPOSE the following user investment query into a goal hierarchy.\n\n"
            f"User Query: \"{query}\"\n\n"
            f"Respond with the JSON goal tree only."
        )
        raw = self.call_llm(prompt, require_json=True)
        tree_dict = self.parse_json(raw)
        return GoalTree(root_goal=tree_dict.get("root_goal", query), sub_goals=tree_dict.get("sub_goals", []))

    def detect_conflicts(self, chain_output: dict, sentiment_output: dict) -> list:
        """Detect conflicting signals between chain risk and sentiment."""
        conflicts = []
        chain_assessments = {a["name"]: a for a in chain_output.get("protocol_assessments", [])}
        sentiment_assessments = {s["protocol_name"]: s for s in sentiment_output.get("protocol_sentiments", [])}

        for name, chain in chain_assessments.items():
            sent = sentiment_assessments.get(name)
            if not sent:
                continue
            # Low risk but negative sentiment
            if chain.get("risk_level") == "LOW" and sent.get("sentiment_score", 0) < -20:
                conflicts.append(f"{name}: Low on-chain risk but negative sentiment (score={sent['sentiment_score']})")
            # High risk but positive sentiment
            if chain.get("risk_level") in ("HIGH", "CRITICAL") and sent.get("sentiment_score", 0) > 20:
                conflicts.append(f"{name}: High on-chain risk but positive sentiment (score={sent['sentiment_score']})")
        return conflicts

    def run(self, query: str, user_profile: dict) -> Recommendation:
        """Execute the full multi-agent pipeline."""
        # Phase 1: Goal Decomposition
        goal_tree = self.decompose_goals(query)

        # Phase 2: Dispatch agents sequentially
        chain_output = self.chain_agent.run()
        sentiment_output = self.sentiment_agent.run()
        risk_output = self.risk_profile_agent.run({
            "user_profile": user_profile,
            "chain_output": chain_output.output_json,
            "sentiment_output": sentiment_output.output_json,
        })

        # Phase 3: Reconciliation & Synthesis
        conflicts = self.detect_conflicts(chain_output.output_json, sentiment_output.output_json)

        synthesis_prompt = (
            f"SYNTHESIZE a unified portfolio recommendation.\n\n"
            f"User Query: \"{query}\"\n"
            f"User Profile: {json.dumps(user_profile)}\n\n"
            f"Goal Tree:\n{json.dumps(goal_tree.to_dict(), indent=2)}\n\n"
            f"ChainAgent Output:\n{json.dumps(chain_output.output_json, indent=2)}\n\n"
            f"SentimentAgent Output:\n{json.dumps(sentiment_output.output_json, indent=2)}\n\n"
            f"RiskProfileAgent Output:\n{json.dumps(risk_output.output_json, indent=2)}\n\n"
            f"Detected Conflicts: {json.dumps(conflicts)}\n\n"
            f"Produce the final Markdown recommendation."
        )
        final_text = self.call_llm(synthesis_prompt)

        return Recommendation(
            goal_tree=goal_tree,
            chain_output=chain_output,
            sentiment_output=sentiment_output,
            risk_output=risk_output,
            final_text=final_text,
        )
