"""
ConMem configuration.

The defaults in this file follow the Methodology.md flow:
aggregate -> admit -> retrieve/trigger -> graph expand -> coordinate -> compose.
Legacy names such as `type_preference_matrix` are kept for compatibility, but
now describe section-level emphasis inside unified cards.
"""
from dataclasses import dataclass, field
import json
import os

from dotenv import load_dotenv


@dataclass
class ConMemConfig:
    # --- Task Identification ---
    task_dedup_threshold: float = 0.95

    # --- Retrieval / Trigger ---
    retrieval_top_k: int = 3
    retrieval_min_relevance: float = 0.30
    retrieval_min_keyword_overlap: float = 0.08
    retrieval_mmr_lambda: float = 0.75
    activation_threshold: float = 0.40
    alpha_relevance: float = 0.30
    alpha_trigger: float = 0.15
    alpha_section_needs: float = 0.20
    alpha_credibility: float = 0.15
    alpha_quality: float = 0.20
    trigger_keyword_weight: float = 0.35
    trigger_similarity_weight: float = 0.65
    enable_kodcode_contract_gate: bool = True
    retrieval_task_needs_keyword_increment: float = 0.20
    retrieval_task_needs_default_uniform_threshold: float = 0.20
    retrieval_section_gate_threshold: float = 0.25
    retrieval_quality_fallback: float = 0.50
    kodcode_off_topic_min_terms: int = 2
    kodcode_off_topic_margin: int = 1
    keyword_token_min_chars: int = 4
    enforce_task_domain_filter: bool = True
    allow_cross_domain_retrieval_fallback: bool = True
    cross_domain_activation_penalty: float = 0.10
    cross_domain_score_penalty: float = 0.15
    allow_cross_domain_graph_edges: bool = False

    # --- Coordination ---
    alignment_epsilon: float = 0.40
    coord_merge_threshold: float = 0.85
    coord_conflict_margin: float = 0.10
    coord_constraint_defer_overlap: float = 0.10

    # --- Admission (Section 4.5) ---
    # Q(c) = λ₁C(c) + λ₂N(c) + λ₃R(c) + λ₄U(c)
    admission_threshold: float = 0.50
    admission_w_reliability: float = 0.30
    admission_w_novelty: float = 0.25
    admission_w_relevance: float = 0.15
    admission_w_utility: float = 0.30
    round_decay_constant: float = 20.0
    admission_consistency_threshold: float = 0.40
    admission_missing_evidence_score: float = 0.20
    admission_density_short_text_chars: int = 80
    admission_density_short_text_penalty: float = 0.50
    admission_density_digit_bonus: float = 0.10
    admission_density_causal_bonus: float = 0.10
    admission_merge_trigger_overlap_threshold: float = 0.18
    admission_merge_section_overlap_threshold: float = 0.18
    reliability_base_constant: float = 0.30
    reliability_coverage_weight: float = 0.35
    reliability_dependency_weight: float = 0.35
    reliability_consistency_weight: float = 0.60
    reliability_evidence_weight: float = 0.40
    utility_coverage_bonus: float = 0.10
    utility_outcome_weight: float = 0.70
    utility_density_weight: float = 0.30
    round_decay_profiles: dict = field(default_factory=lambda: {
        "kodcode": 217.0,
        "triviaqa": 651.0,
        "popqa": 760.0,
        "pddl": 12.0,
    })

    # --- Serialization / Compose ---
    token_budget: int = 1600
    serializer_max_cards_per_section: int = 2
    serializer_max_card_chars: int = 450
    serializer_include_evidence: bool = False
    disable_factual_qa_evaluator_memory: bool = True
    serializer_compact_card_chars: int = 240
    serializer_full_section_budget: int = 2
    serializer_compact_section_budget: int = 1
    serializer_full_trigger_chars: int = 160
    serializer_compact_trigger_chars: int = 100
    serializer_full_when_chars: int = 180
    serializer_compact_when_chars: int = 120
    serializer_full_check_chars: int = 220
    serializer_compact_check_chars: int = 140
    serializer_warning_chars: int = 160
    serializer_card_id_chars: int = 8
    serializer_trigger_examples: int = 2

    # --- Card Graph ---
    graph_similarity_threshold: float = 0.60
    graph_candidate_top_k: int = 10
    graph_walk_hops: int = 2
    graph_walk_weight_threshold: float = 0.40
    graph_expansion_max_cards: int = 6
    graph_heuristic_default_weight: float = 0.40
    graph_heuristic_conflict_overlap: float = 0.20
    graph_heuristic_support_similarity: float = 0.85
    graph_heuristic_support_overlap: float = 0.45
    graph_heuristic_satisfies_overlap: float = 0.20
    graph_heuristic_constrains_overlap: float = 0.20
    graph_constraint_activation_overlap: float = 0.10
    graph_relation_default_weight: float = 0.50

    # --- Compression & Post-Commit Merge ---
    trajectory_max_tokens: int = 4096
    commit_merge_threshold: float = 0.90
    cross_task_merge_threshold: float = 0.84
    cross_task_signature_min_overlap: int = 2
    # Throttle post_commit_merge: only run every K completed tasks (1 = every task).
    admission_post_commit_merge_every_k_tasks: int = 20
    extract_existing_card_examples: int = 5
    extract_existing_section_preview_chars: int = 120
    extract_existing_summary_preview_chars: int = 160
    summary_preview_section_chars: int = 220
    summary_preview_evidence_chars: int = 160
    trigger_section_priority: dict = field(default_factory=lambda: {
        "plan": 3.0,
        "eval": 2.0,
        "state": 1.0,
    })
    trigger_min_sentence_chars: int = 12
    trigger_sentence_clip_chars: int = 180
    trigger_sentence_length_norm_chars: int = 100
    trigger_max_semantics: int = 6

    # --- Graph Explosion Safeguard ---
    max_graph_nodes: int = 10000

    # --- Ablation Flags ---
    enable_coordination: bool = True
    enable_graph_expansion: bool = True
    enable_failure_reflection: bool = True
    enable_failure_admission: bool = True
    skip_uninformative_failed_trajectories: bool = True

    # --- Graph relation: heuristic-first (skip LLM unless ambiguous) ---
    graph_relation_heuristic_first: bool = True
    graph_relation_ambiguous_range: tuple = (0.30, 0.70)

    # --- Benchmark-specific task needs presets ---
    # Overrides keyword-based task needs analysis when task_domain is set
    task_needs_presets: dict = field(default_factory=lambda: {
        "triviaqa": {"state": 0.5333, "plan": 0.1333, "exec": 0.2000, "eval": 0.1333},
        "popqa":    {"state": 0.5333, "plan": 0.1333, "exec": 0.2000, "eval": 0.1333},
        "kodcode":  {"state": 0.0909, "plan": 0.2727, "exec": 0.4091, "eval": 0.2273},
        "pddl":     {"state": 0.3043, "plan": 0.3478, "exec": 0.2174, "eval": 0.1304},
    })

    # --- Section-aware role allocation ---
    role_budget_allocation: dict = field(default_factory=lambda: {
        "planner": {"state": 0.35, "plan": 0.35, "exec": 0.10, "eval": 0.20},
        "executor": {"state": 0.15, "plan": 0.20, "exec": 0.40, "eval": 0.25},
        "evaluator": {"state": 0.20, "plan": 0.15, "exec": 0.30, "eval": 0.35},
        "default": {"state": 0.25, "plan": 0.25, "exec": 0.25, "eval": 0.25},
    })

    type_preference_matrix: dict = field(default_factory=lambda: {
        "planner": {"state": 0.9, "plan": 0.9, "exec": 0.3, "eval": 0.6},
        "executor": {"state": 0.5, "plan": 0.6, "exec": 0.9, "eval": 0.8},
        "evaluator": {"state": 0.5, "plan": 0.4, "exec": 0.7, "eval": 0.9},
        "default": {"state": 0.6, "plan": 0.6, "exec": 0.6, "eval": 0.6},
    })

    utility_table: dict = field(default_factory=lambda: {
        ("eval", "success"): 0.9,
        ("eval", "partial"): 0.8,
        ("eval", "failure"): 0.95,
        ("plan", "success"): 0.8,
        ("plan", "partial"): 0.6,
        ("plan", "failure"): 0.85,
        ("state", "success"): 0.6,
        ("state", "partial"): 0.5,
        ("state", "failure"): 0.65,
        ("exec", "success"): 0.5,
        ("exec", "partial"): 0.4,
        ("exec", "failure"): 0.55,
        ("card", "success"): 0.85,
        ("card", "partial"): 0.65,
        ("card", "failure"): 0.90,
    })

    credibility_table: dict = field(default_factory=lambda: {
        "success": 1.0,
        "partial": 0.8,
        "failure": 0.9,  # Failure memories are valuable negative knowledge, but not perfect evidence
    })
    credibility_reflection_floor: float = 0.70
    credibility_reflection_weight: float = 0.30

    type_dependency_chain: list = field(default_factory=lambda: [
        ("state", "plan"),
        ("plan", "exec"),
        ("exec", "eval"),
    ])

    # --- LLM & Embedding Configuration ---
    llm_api_key: str = ""
    llm_base_url: str = "http://localhost:8100/v1"
    llm_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    embed_api_key: str = ""
    embed_base_url: str = "http://localhost:8002/v1"
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embed_device: str = ""

    # --- LLM Call Settings ---
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096
    llm_retry_count: int = 1
    llm_timeout: float = 120.0
    llm_max_input_chars: int = 120000
    trajectory_chars_per_token_estimate: int = 4

    # --- Factual-QA Evidence Compaction ---
    qa_search_url: str = "http://127.0.0.1:8000/retrieve"
    qa_search_topk: int = 3
    qa_search_timeout_seconds: float = 30.0
    qa_compaction_max_total_chars: int = 1800
    qa_compaction_max_doc_chars: int = 520
    qa_compaction_title_chars: int = 80
    qa_compaction_doc_slack_chars: int = 40
    qa_compaction_remaining_floor_chars: int = 120
    qa_compaction_max_chunks_per_source: int = 1
    qa_exchange_query_chars: int = 120
    qa_exchange_answer_chars: int = 240
    qa_exchange_min_info_budget_chars: int = 220
    qa_exchange_max_info_blocks: int = 2
    qa_actor_exchange_total_chars: int = 1600
    qa_actor_exchange_info_chars: int = 900
    qa_actor_max_search_iters: int = 4  # Search-R1 interaction budget B=4; LatentMem follows same protocol
    qa_critic_exchange_total_chars: int = 900
    qa_critic_exchange_info_chars: int = 500
    qa_proxy_exchange_total_chars: int = 1400
    qa_proxy_exchange_info_chars: int = 800
    qa_search_profiles: dict = field(default_factory=lambda: {
        "triviaqa": {
            "search_url": "http://127.0.0.1:8000/retrieve",
        },
        "popqa": {
            "search_url": "http://127.0.0.1:8000/retrieve",
        },
    })

    @property
    def role_section_weights(self) -> dict:
        return self.type_preference_matrix

    def apply_overrides(self, overrides: dict):
        config_fields = set(self.__dataclass_fields__)
        for key, value in overrides.items():
            if key in config_fields:
                if key == "graph_relation_ambiguous_range" and isinstance(value, list):
                    value = tuple(value)
                setattr(self, key, value)

    def sync_runtime_settings(self):
        from .schema import configure_schema_preview

        configure_schema_preview(
            section_chars=self.summary_preview_section_chars,
            evidence_chars=self.summary_preview_evidence_chars,
        )

    def token_regex(self) -> str:
        return rf"\b[a-zA-Z_]{{{max(1, int(self.keyword_token_min_chars))},}}\b"

    def factual_qa_retriever_kwargs(self) -> dict:
        return self.factual_qa_retriever_kwargs_for_domain()

    def get_round_decay_constant(self, task_domain: str | None = None) -> float:
        if task_domain and task_domain in self.round_decay_profiles:
            return float(self.round_decay_profiles[task_domain])
        return float(self.round_decay_constant)

    def apply_task_domain_profile(self, task_domain: str | None):
        if not task_domain:
            return
        self.round_decay_constant = self.get_round_decay_constant(task_domain)

    def factual_qa_retriever_kwargs_for_domain(self, task_domain: str | None = None) -> dict:
        profile = self.qa_search_profiles.get(task_domain, {}) if task_domain else {}
        return {
            "search_url": profile.get("search_url", self.qa_search_url),
            "topk": profile.get("topk", self.qa_search_topk),
            "timeout_seconds": profile.get("timeout_seconds", self.qa_search_timeout_seconds),
            "max_doc_chars": self.qa_compaction_max_doc_chars,
            "max_total_chars": self.qa_compaction_max_total_chars,
            "title_chars": self.qa_compaction_title_chars,
            "doc_slack_chars": self.qa_compaction_doc_slack_chars,
            "remaining_floor_chars": self.qa_compaction_remaining_floor_chars,
            "max_chunks_per_source": self.qa_compaction_max_chunks_per_source,
        }

    @classmethod
    def from_env(cls, env_path: str = None) -> "ConMemConfig":
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        def _first_nonempty_env(*keys: str, default: str = "") -> str:
            for key in keys:
                value = os.getenv(key)
                if value:
                    return value
            return default

        config = cls()
        config.llm_api_key = _first_nonempty_env("LLM_API_KEY", default="")
        config.llm_base_url = _first_nonempty_env("LLM_BASE_URL", default=config.llm_base_url)
        config.llm_model = _first_nonempty_env("LLM_MODEL", default=config.llm_model)
        config.embed_api_key = _first_nonempty_env("CONMEM_EMBED_API_KEY", "EMBED_API_KEY", default="")
        config.embed_base_url = _first_nonempty_env(
            "CONMEM_EMBED_BASE_URL",
            "EMBED_BASE_URL",
            default=config.embed_base_url,
        )
        config.embed_model = _first_nonempty_env(
            "CONMEM_EMBED_MODEL",
            "EMBED_MODEL",
            default=config.embed_model,
        )
        config.embed_device = _first_nonempty_env("CONMEM_EMBED_DEVICE", default=config.embed_device)

        if os.getenv("MEMORY_ADMISSION_THRESHOLD"):
            config.admission_threshold = float(os.getenv("MEMORY_ADMISSION_THRESHOLD"))
        if os.getenv("ALIGNMENT_EPSILON"):
            config.alignment_epsilon = float(os.getenv("ALIGNMENT_EPSILON"))
        if os.getenv("TOPK"):
            config.retrieval_top_k = int(os.getenv("TOPK"))
        if os.getenv("CONMEM_ACTIVATION_THRESHOLD"):
            config.activation_threshold = float(os.getenv("CONMEM_ACTIVATION_THRESHOLD"))
        if os.getenv("CONMEM_RETRIEVAL_MIN_RELEVANCE"):
            config.retrieval_min_relevance = float(os.getenv("CONMEM_RETRIEVAL_MIN_RELEVANCE"))
        if os.getenv("CONMEM_RETRIEVAL_MIN_KEYWORD_OVERLAP"):
            config.retrieval_min_keyword_overlap = float(os.getenv("CONMEM_RETRIEVAL_MIN_KEYWORD_OVERLAP"))
        if os.getenv("CONMEM_TOKEN_BUDGET"):
            config.token_budget = int(os.getenv("CONMEM_TOKEN_BUDGET"))
        if os.getenv("CONMEM_MAX_CARDS_PER_SECTION"):
            config.serializer_max_cards_per_section = int(os.getenv("CONMEM_MAX_CARDS_PER_SECTION"))
        if os.getenv("CONMEM_GRAPH_EXPANSION_MAX_CARDS"):
            config.graph_expansion_max_cards = int(os.getenv("CONMEM_GRAPH_EXPANSION_MAX_CARDS"))
        if os.getenv("CONMEM_ENABLE_GRAPH_EXPANSION") is not None:
            config.enable_graph_expansion = os.getenv("CONMEM_ENABLE_GRAPH_EXPANSION") == "1"
        if os.getenv("CONMEM_ENABLE_COORDINATION") is not None:
            config.enable_coordination = os.getenv("CONMEM_ENABLE_COORDINATION") == "1"
        if os.getenv("CONMEM_ENABLE_FAILURE_REFLECTION") is not None:
            config.enable_failure_reflection = os.getenv("CONMEM_ENABLE_FAILURE_REFLECTION") == "1"
        if os.getenv("CONMEM_ENABLE_FAILURE_ADMISSION") is not None:
            config.enable_failure_admission = os.getenv("CONMEM_ENABLE_FAILURE_ADMISSION") == "1"
        if os.getenv("CONMEM_ENABLE_KODCODE_CONTRACT_GATE") is not None:
            config.enable_kodcode_contract_gate = os.getenv("CONMEM_ENABLE_KODCODE_CONTRACT_GATE") == "1"
        if os.getenv("CONMEM_DISABLE_FACTUAL_QA_EVALUATOR_MEMORY") is not None:
            config.disable_factual_qa_evaluator_memory = os.getenv("CONMEM_DISABLE_FACTUAL_QA_EVALUATOR_MEMORY") == "1"
        profile_path = os.getenv("CONMEM_HYPERPARAM_PROFILE")
        if profile_path and os.path.exists(profile_path):
            with open(profile_path, "r", encoding="utf-8") as f:
                overrides = json.load(f)
            if isinstance(overrides, dict):
                config.apply_overrides(overrides)

        # Explicit environment overrides should win over profile files. This
        # lets smoke tests and ablations relax retrieval gates without editing
        # the calibrated profile JSON.
        if os.getenv("TOPK"):
            config.retrieval_top_k = int(os.getenv("TOPK"))
        if os.getenv("CONMEM_ACTIVATION_THRESHOLD"):
            config.activation_threshold = float(os.getenv("CONMEM_ACTIVATION_THRESHOLD"))
        if os.getenv("CONMEM_RETRIEVAL_MIN_RELEVANCE"):
            config.retrieval_min_relevance = float(os.getenv("CONMEM_RETRIEVAL_MIN_RELEVANCE"))
        if os.getenv("CONMEM_RETRIEVAL_MIN_KEYWORD_OVERLAP"):
            config.retrieval_min_keyword_overlap = float(os.getenv("CONMEM_RETRIEVAL_MIN_KEYWORD_OVERLAP"))
        if os.getenv("CONMEM_TOKEN_BUDGET"):
            config.token_budget = int(os.getenv("CONMEM_TOKEN_BUDGET"))
        if os.getenv("CONMEM_MAX_CARDS_PER_SECTION"):
            config.serializer_max_cards_per_section = int(os.getenv("CONMEM_MAX_CARDS_PER_SECTION"))
        if os.getenv("CONMEM_GRAPH_EXPANSION_MAX_CARDS"):
            config.graph_expansion_max_cards = int(os.getenv("CONMEM_GRAPH_EXPANSION_MAX_CARDS"))

        config.sync_runtime_settings()
        return config
