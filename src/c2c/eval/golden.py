"""The golden values — every number the paper publishes, as published.

Source: Fu et al., *Cache-to-Cache: Direct Semantic Communication Between
Large Language Models*, ICLR 2026, arXiv:2510.03215v2. Each entry records
the table it came from; entries the extractor could not deliver are
collected in :data:`UNAVAILABLE` with their reason, per the release policy
of this package: no invented numbers, no transcriptions left to the
imagination.

The tolerance for the main results is ±0.5 % (absolute, spec §4.2
acceptance); the ablation ordering of Table 8 is asserted as a *floor*:
any build whose ordering of the three columns is not
``Project < +Fuse < +Gate`` on the average fails review.
"""

from __future__ import annotations

__all__ = ["TABLES", "FIGURES", "UNAVAILABLE", "TOLERANCE", "PROTOCOL", "TABLE_KEYWORDS"]

#: The evaluation protocol, verbatim from §4.1 / §5.
PROTOCOL = {
    "zero_shot": True,
    "greedy": True,
    "temperature": 0.0,
    "max_response_tokens": 64,
    "communication_tokens": 256,
    "longbench_max_out": 2048,
    "seed": 42,
    "batch_size": 1,
    "device": "single NVIDIA A100",
}

#: absolute tolerance, in accuracy points, per the acceptance clause
TOLERANCE = 0.5

RECEIVER = "Qwen3-0.6B"
BENCHMARKS_ORDER = ("MMLU-Redux", "ARC-C", "OpenBookQA", "C-Eval")

TABLES: dict[str, dict] = {
    # -- abstract claims, §1 ------------------------------------------------------------
    "abstract": {
        "gain_over_individual_pct": (6.4, 14.2),      # "6.4-14.2 % higher average"
        "gain_over_t2t_pct": (3.1, 5.4),               # "3.1-5.4 %"
        "average_speedup": 2.5,                        # "2.5× speedup"
        "fusion_overhead_ms": 90.0,                    # Table 3: "*Includes 90 ms"
    },

    # -- Table 1: cache-enrichment oracle (§3.2.1) --------------------------------------
    "table01": {
        "columns": ("method", "cache_len", "enriched", "accuracy"),
        "rows": [
            ("Direct", "X", False, 58.42),
            ("Few-shot", "E+X", True, 63.39),
            ("Oracle", "X", True, 62.34),
        ],
    },

    # -- Table 2: effective rank, before/after (§4.5) -----------------------------------
    "table02": {
        "columns": ("kind", "sharer", "receiver", "c2c"),
        "rows": [
            ("K", 539.0, 388.0, 395.0),
            ("V", 689.0, 532.0, 560.0),
        ],
        "claim": "increase on the receiver side after fusion",
    },

    # -- Table 3: token counts and inference time, MMLU-Redux (§4.2) -------------------
    "table03": {
        "pair": ("Qwen2.5-0.5B-Instruct → Qwen3-0.6B"),
        "columns": ("metric", "receiver_only", "sharer_only",
                   "t2t_sharer", "t2t_receiver", "c2c_sharer", "c2c_receiver"),
        "rows": [
            ("input_tokens", 170, 187, 103, 332, 170, 170),
            ("output_tokens", 11, 19, 80, 10, 0, 12),
            ("prefill_ms", 27, 20, 21, 32, 20 + 90, 27),   # 90 ms KV fusion
            ("decode_ms", 281, 326, 1312, 231, 0, 308),
            ("total_ms", 308, 346, 1596, None, 445, None),
        ],
        "speedup_t2t_to_c2c": 1596 / 445,
    },

    # -- Table 4 + §4.2: main results, receiver fixed (Qwen3-0.6B) ----------------------
    # accuracy (%), time (s), the columns of the table as printed
    "table04": {
        "receiver": RECEIVER,
        "benchmarks": BENCHMARKS_ORDER,
        "receiver_only": {"MMLU-Redux": 35.53, "OpenBookQA": 39.20,
                        "ARC-C": 41.04, "C-Eval": 32.04},
        "receiver_only_time": {"MMLU-Redux": 0.29, "OpenBookQA": 0.27,
                            "ARC-C": 0.29, "C-Eval": 0.26},
        "sharers": {
            "Qwen2.5-0.5B-Instruct": {
                "sharer_only": {"MMLU-Redux": 38.42, "OpenBookQA": 45.60,
                             "ARC-C": 42.09, "C-Eval": 40.21},
                "sharer_only_time": {"MMLU-Redux": 0.34, "OpenBookQA": 0.35,
                                  "ARC-C": 0.39, "C-Eval": 0.31},
                "t2t": {"MMLU-Redux": 41.03, "OpenBookQA": 44.00,
                      "ARC-C": 49.48, "C-Eval": 35.88},
                "t2t_time": {"MMLU-Redux": 1.52, "OpenBookQA": 0.81,
                          "ARC-C": 1.00, "C-Eval": 1.51},
                "c2c": {"MMLU-Redux": 42.92, "OpenBookQA": 52.60,
                      "ARC-C": 54.52, "C-Eval": 41.77},
                "c2c_time": {"MMLU-Redux": 0.40, "OpenBookQA": 0.30,
                          "ARC-C": 0.36, "C-Eval": 0.34},
                "routing": {"MMLU-Redux": 35.58, "OpenBookQA": 40.80,
                         "ARC-C": 40.70, "C-Eval": 34.61},
                "routing_time": {"MMLU-Redux": 0.27, "OpenBookQA": 0.29,
                             "ARC-C": 0.29, "C-Eval": 0.26},
                "gain_over_individual": 11.00,
                "gain_over_t2t": 5.36,
                "speedup": 3.46,
            },
            "Llama3.2-1B": {
                "sharer_only": {"MMLU-Redux": 32.30, "OpenBookQA": 32.60,
                             "ARC-C": 33.57, "C-Eval": 31.31},
                "sharer_only_time": {"MMLU-Redux": 0.06, "OpenBookQA": 0.07,
                                  "ARC-C": 0.07, "C-Eval": 0.04},   # App. A.4.3, the fast one
                "t2t": {"MMLU-Redux": 43.32, "OpenBookQA": 41.20,
                      "ARC-C": 50.00, "C-Eval": 35.27},
                "t2t_time": {"MMLU-Redux": 0.75, "OpenBookQA": 0.70,
                          "ARC-C": 0.70, "C-Eval": 0.71},
                "c2c": {"MMLU-Redux": 44.42, "OpenBookQA": 47.80,
                      "ARC-C": 53.39, "C-Eval": 40.77},
                "c2c_time": {"MMLU-Redux": 0.50, "OpenBookQA": 0.43,
                          "ARC-C": 0.47, "C-Eval": 0.49},
                "routing": {"MMLU-Redux": 33.38, "OpenBookQA": 36.40,
                         "ARC-C": 37.22, "C-Eval": 31.92},
                "routing_time": {"MMLU-Redux": 0.18, "OpenBookQA": 0.17,
                             "ARC-C": 0.18, "C-Eval": 0.15},
                "gain_over_individual": 9.64,
                "gain_over_t2t": 4.15,
                "speedup": 1.51,
            },
            "Qwen3-4B-Base": {
                "sharer_only": {"MMLU-Redux": 1.03, "OpenBookQA": 2.20,        # the base
                             "ARC-C": 1.48, "C-Eval": 5.65},                   # ignores the
                "sharer_only_time": {"MMLU-Redux": 2.06, "OpenBookQA": 1.98,   # instructions
                                  "ARC-C": 2.06, "C-Eval": 2.02},
                "t2t": {"MMLU-Redux": 43.87, "OpenBookQA": 46.40,
                      "ARC-C": 53.91, "C-Eval": 38.92},
                "t2t_time": {"MMLU-Redux": 7.54, "OpenBookQA": 5.08,           # excessively long
                          "ARC-C": 6.56, "C-Eval": 3.59},                      # t2t, as printed
                "c2c": {"MMLU-Redux": 43.95, "OpenBookQA": 53.20,
                      "ARC-C": 55.39, "C-Eval": 42.79},
                "c2c_time": {"MMLU-Redux": 0.45, "OpenBookQA": 0.34,
                          "ARC-C": 0.40, "C-Eval": 0.39},
                "routing": {"MMLU-Redux": 16.39, "OpenBookQA": 22.20,
                         "ARC-C": 19.65, "C-Eval": 15.10},
                "routing_time": {"MMLU-Redux": 0.28, "OpenBookQA": 0.27,
                             "ARC-C": 0.28, "C-Eval": 0.26},
                "gain_over_individual": 11.88,
                "gain_over_t2t": 3.06,
                "speedup": 14.41,
            },
        },
        "routing_note": "query-level routing never exceeds the better of the pair",
    },

    # -- Table 5: scaling with sequence length, LongBenchV1 (§4.3) -----------------------
    "table05": {
        "columns": ("length", "receiver", "sharer", "t2t", "c2c"),
        "rows": [
            ("0-4k", 30.52, 24.94, 33.46, 37.31),
            ("4-8k", 26.03, 23.18, 29.70, 34.01),
            ("8k+", 25.99, 16.44, 25.64, 30.72),
        ],
        "claim": "C2C outperforms T2T across all sequence-length intervals",
    },

    # -- Table 6: sources of improvement (§4.4) -----------------------------------------
    "table06": {
        "columns": ("setting", "params", "OpenBookQA", "ARC-C", "MMLU-Redux", "C-Eval"),
        "rows": [
            ("Single", "596M", 45.80, 47.65, 36.81, 35.81),
            ("Identical", "529M", 50.60, 52.52, 42.17, 40.34),
            ("C2C", "478M", 52.60, 54.52, 42.92, 41.77),
        ],
        "claim": "Single < Identical < C2C consistently (higher accuracy)",
    },

    # -- Table 7: further pairs, heterogeneous and swap (§4.3) ---------------------------
    "table07": {
        "columns": ("group", "receiver", "sharer", "receiver_acc", "sharer_acc",
                   "t2t_acc", "c2c_acc", "receiver_time", "t2t_time", "c2c_time"),
        "rows": [
            ("heterogeneous", "Qwen3-0.6B", "Gemma3-1B", 35.53, 31.75, 41.35, 45.90,
           0.29, 1.04, 0.30),
            ("heterogeneous", "Qwen3-0.6B", "Qwen2.5-Math-1.5B", 35.53, 39.86,
           43.71, 46.13, 0.29, 6.60, 0.27),
            ("heterogeneous", "Qwen3-0.6B", "Qwen2.5-Coder-0.5B", 35.53, 25.09,
           39.74, 46.89, 0.29, 1.59, 0.27),
            ("swap", "Qwen2.5-0.5B", "Qwen3-0.6B", 38.42, 35.53, 32.12, 43.47,
           0.34, 0.98, 0.21),
            ("swap", "Qwen3-0.6B", "Qwen2.5-0.5B", 35.53, 38.42, 41.03, 46.50,
           0.29, 1.52, 0.26),
        ],
        "claims": {
            "heterogeneous_avg_gain_over_t2t": 8.59,
            "swap_c2c_gain": 5.05,
            "swap_t2c_change": -6.30,
        },
    },

    # -- Table 8: ablation of the fuser, the ABLATION FLOOR (§4.4) -----------------------
    "table08": {
        "columns": ("method", "MMLU-Redux", "ARC-C", "OpenBookQA", "C-Eval", "Average"),
        "rows": [
            ("Project", 20.01, 19.57, 21.80, 21.41, 20.70),
            ("+Fuse", 43.36, 51.65, 47.60, 36.91, 44.88),
            ("+Gate", 42.92, 54.52, 52.60, 41.77, 47.95),
        ],
        "floor": {
            "project_average": 20.70,
            "fuse_delta_over_project": 24.18,
            "gate_delta_over_fuse": 3.07,
            "ordering": ("Project", "+Fuse", "+Gate"),
        },
    },
}

#: Appendix and beyond: the numbers the specification cites but the PDF
#: text extraction did not deliver as machine-readable tables. They are
#: transcribed from the specification's own quotations of the paper
#: (C2C-SPEC.md, §2.2 and §3) — where the two disagree, the paper wins;
#: where the paper is unread here, the value is held unavailable.
TABLES["table12"] = {
    "claim": "training cost at 300 steps on one A100",
    "gpu_hours": 9.0, "at_steps": 300,
}
TABLES["table13"] = {
    "claim": "many-to-one: two Sharers, one Receiver",
    "receivers": 1, "sharers": 2, "average_accuracy": 64.60,
}
TABLES["table15"] = {
    "claim": "agentic flow, interpreter+solver, GSM8K",
    "t_c2c_accuracy": 78.01, "benchmark": "GSM8K",
}

#: Figures 3–13 as named in the paper; the pixel data of the plots was
#: not part of the text extraction. Every one of them joins the fellowship
#: of the unavailable, and the golden suite skips its test with this reason.
FIGURES: dict[str, str] = {
    f"figure{n:02}": f"Figure {n} (§/App. — plot data not machine-readable"
                    f" in the arXiv:2510.03215v2 text extraction)"
    for n in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
}

#: Table 9 (C2C-C comparison) and Table 14 (many-to-many) join the same
#: fellowship of unavailable entries.
UNAVAILABLE: dict[str, str] = {
    "table09": "C2C-C comparison (App. A.1.3): set as image; not extracted",
    "table10": "not referenced by the specification; not extracted",
    "table11": "not referenced by the specification; not extracted",
    "table14": "multi-sharer/multi-receiver budgets (App. A.5.2): image; not extracted",
    **{k: v for k, v in FIGURES.items()},
}

#: Table N of the paper, as the eye of ``c2c eval --table N`` selects the
#: tests that defend it. The names are the node names of the golden suite;
#: one keyword string each, joined by the logic of pytest's ``-k``.
TABLE_KEYWORDS: dict[int, str] = {
    1: "table_1_the_enrichment_oracle",
    2: "table_2_the_effective_rank_increases",
    3: "table_3_the_fusion_costs",
    4: ("c2c_beats or gains_are_the_published or deltas_are_consistent "
        "or table_4_reproduced"),
    5: "table_5_wins",
    6: "table_6_the_sources",
    7: "table_7_the_pairs",
    8: "ablations_is_the_floor",
}
