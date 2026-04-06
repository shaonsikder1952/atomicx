# AtomicX

**Understanding the butterfly effect at atomic level.**

AtomicX decomposes reality into atomic variables, traces how microscopic changes cascade through causal relationships across multi-layer validation, and predicts how small shifts produce large outcomes. Self-evolving architecture learns which atomic patterns precede events through continuous empirical verification.

**Current implementation**: Financial markets (crypto, equities). **Architecture**: Domain-agnostic (designed for any measurable reality—weather, health, supply chains, social systems).

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research-orange.svg)](https://github.com/shaonsikder1952/atomicx)

---

## System Architecture

AtomicX decomposes observable reality into atomic variables, discovers structural relationships using causal inference algorithms (NOTEARS, PC, Granger), tracks empirical patterns through outcome verification, and validates predictions across independent analytical layers. The system autonomously evolves at three levels: agents, strategies, and architecture.

**Core methodology:**
- **Variable decomposition**: Break reality into atomic measurable units (domain-specific: RSI for markets, temperature for weather, heart rate for health)
- **Causal discovery** (observational): NOTEARS/PC/Granger algorithms identify directed acyclic graph structure
- **Pattern tracking** (empirical): Historical co-occurrence analysis with continuous accuracy monitoring
- **Multi-layer validation**: Independent layers vote, fusion gate requires consensus threshold
- **Self-evolution**: Agents with poor performance auto-disable or invert to contrarian signals

**Universal domains** (any measurable reality):
- **PHYSICAL**: Tangible measurements (order flow, temperature, inventory, pressure)
- **BIOLOGICAL**: Growth/decay patterns (network effects, viral spread, cell behavior)
- **ECONOMIC**: Resource allocation (price, demand, supply, costs)
- **SOCIAL**: Community signals (news, sentiment, collective behavior)
- **PSYCHOLOGICAL**: Emotions, narratives, perception shifts
- **BEHAVIORAL**: Observed actions (trading patterns, movement, choices)
- **TEMPORAL**: Time-based patterns (seasonality, cycles, rhythms)

**Current implementation:**
- **Validated**: Financial markets (crypto via Binance, equities via Yahoo Finance)
- **Variables**: 47 atomic measurements (momentum, volatility, volume, microstructure, etc.)
- **Layers**: 14 independent validation layers
- **Agents**: 62-agent hierarchy (atomic → groups → super groups)
- **Limitation**: Pattern discovery trained exclusively on financial data

**Status**: Research-stage. Architecture designed for universal applicability. No published benchmarks vs baselines. Requires domain-specific validation before production deployment.

---

## Technical Stack

**Execution**: Python 3.11+, async/await event loop  
**Data**: Polars (vectorized), Numba (JIT), TimescaleDB (time-series), Redis (cache), Kafka (streams)  
**Causal inference**: NOTEARS, PC Algorithm (causal-learn), Granger Causality  
**Agents**: Hierarchical multi-agent system (atomic agents → group leaders → domain super groups)  
**Simulation**: Heterogeneous agent swarm with Numba JIT compilation  
**Intelligence**: CRUCIX OSINT (multi-source intelligence), Playwright browser automation, LLM failover (Anthropic/Bedrock/Ollama)  
**Memory**: PostgreSQL + Qdrant (vector), Wiki (markdown knowledge base)  
**Dashboard**: FastAPI + SSE real-time updates

**Current scale** (financial markets): 47 variables, 14 layers, 62 agents, 500-agent swarm, 27 OSINT sources

**Key integrations:**
- [OpenClaw](https://github.com/openclaw/openclaw): LLM failover, browser pool, lane queues
- [Karpathy LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f): Persistent knowledge system
- [CRUCIX](https://github.com/calesthio/Crucix): OSINT intelligence terminal (integrated as submodule)

---

## Multi-Layer Validation Stack

**Current implementation** (14 layers for financial markets — layer count varies by domain complexity):

**Data ingestion and decomposition (Layers 1-3):**
1. **Sensory ingestion**: Domain-specific data streams (current: WebSocket/REST for financial data)
2. **Context intelligence**: Environmental signals (current: CRUCIX OSINT with 27 sources)
3. **Variable engine**: Atomic measurements computed via Polars (current: 47 financial variables—momentum, volatility, volume, microstructure, etc.)

**Pattern discovery (Layers 4-5):**
4. **Causal discovery**: NOTEARS/PC/Granger → directed acyclic graph with edge weights
5. **Actor/entity analysis**: Model participant incentives and behavior (current: whale/market maker/retail for markets; adaptable to any domain with agents)

**Multi-agent validation (Layers 6-9):**
6. **Swarm simulation**: Heterogeneous agents with Numba JIT (current: 500 market participants; scales to domain)
7. **Agent hierarchy**: Atomic agents → category groups → domain super groups (architecture scales with variable count)
8. **Meta-reasoning**: Reflector examines decision logic for contradictions and blind spots
9. **Narrative analysis**: Context-event correlation (current: news-price; generalizes to any event-outcome relationship)

**Decision and evolution (Layers 10-14):**
10. **Fusion gate**: Weighted vote aggregation, requires consensus threshold for signal validity
11. **Evolution engine**: Per-prediction performance tracking, auto-prune poor performers, auto-invert consistently wrong agents, strategy mutation
12. **Execution**: Domain-specific action layer (current: position sizing, order routing; adaptable to any decision execution)
13. **Memory persistence**: PostgreSQL (structured), Qdrant (vector), Wiki (knowledge base)
14. **Error recovery**: LLM failover, data source reconnection, graceful degradation

---

## The Complete Flow: Observable Reality → Prediction

**Concrete example** (current financial market implementation):

**Scenario**: BTC drops 3% in 15 minutes (from $67,234 → $65,123)

This section traces how observable data transforms through the system's layers into a final prediction. The same architecture applies to any domain—this example uses financial markets because that's the current validated implementation.

### Layer 1: Sensory Data Ingestion

**Input**: Real-time market data streams

```
Binance WebSocket → Raw data
├─ Trades: price=$65,123, volume=2.3M BTC in 15min
├─ Order book: 450 BTC sell orders at $65,100-65,200
├─ Order book: 220 BTC buy orders at $64,800-65,000
├─ Liquidations: $12M long positions liquidated
└─ CVD: Cumulative volume delta = -450 BTC (net selling)
```

**Output**: Timestamped raw data stored in TimescaleDB

---

### Layer 2: News Intelligence Scanner

**Input**: RSS feeds from Reuters, Bloomberg, CoinDesk

```
Scanning for market-moving events:
├─ Keywords: "BTC", "Bitcoin", "SEC", "ETF", "regulation"
├─ Significance scoring: 0-1 scale
└─ Result: No major news events (significance < 0.3)
```

**Output**: `news_sentiment = "neutral"`, `significance = 0.1`

---

### Layer 3: Variable Computation Engine

**Input**: Raw price/volume data from Layer 1

**Processing**: Polars vectorized operations compute 47 variables

```
Variables computed from the 3% drop:

Momentum (8 variables):
├─ momentum_rsi_14: 72 → 58 (momentum weakening)
├─ momentum_macd_line: 0.023 → -0.018 (bearish crossover)
└─ momentum_stoch_rsi: 0.89 → 0.42 (rapid decline)

Volatility (9 variables):
├─ volatility_bb_bandwidth: 0.008 → 0.012 (expansion)
├─ volatility_atr_14: 890 → 1,240 (increased volatility)
└─ volatility_bb_percent_b: 0.92 → 0.18 (lower band breach)

Microstructure (6 variables):
├─ microstructure_ob_imbalance: -0.05 → -0.15 (strong sell pressure)
├─ microstructure_cvd: -120 → -450 (net selling acceleration)
└─ microstructure_bid_volume: 8.2M → 4.3M (bid liquidity evaporated)

Volume (4 variables):
├─ volume_rel_volume: 1.2x → 3.4x (high activity)
└─ volume_obv: 12.4M → 8.9M (distribution)

Trend (9 variables):
├─ trend_ema_9: $66,800 → $65,400 (downtrend)
└─ trend_vwap: $66,950 (price below VWAP = bearish)

Leverage (3 variables):
└─ leverage_funding_rate: +0.02% → -0.01% (longs → shorts)

... (47 variables total)
```

**Output**: 47 timestamped variable values → PostgreSQL

---

### Layer 4: Causal Pattern Discovery

**Input**: Current variable states from Layer 3

**Processing**: Search causal DAG for matching patterns

```
Pattern matching:
┌──────────────────────────────────────────────────┐
│ Pattern: RSI_drop_OB_bearish                     │
├──────────────────────────────────────────────────┤
│ Causal structure (DAG):                          │
│   RSI_high → RSI_drop ──┐                        │
│   OB_imbalance_negative ├──→ price_drop_4h       │
│   Volume_spike ─────────┘                        │
│                                                  │
│ Conditions:                                      │
│   IF RSI > 70 initially                          │
│   AND RSI drops >10 points in 15min             │
│   AND OB_imbalance < -0.10                       │
│   AND volume > 2x average                        │
│   THEN continued drop within 4h                  │
│                                                  │
│ Historical performance:                          │
│   Accuracy: 68% (n=127 samples)                  │
│   Last reinforced: 2024-03-15 (8 months ago)     │
│   P-value: 0.003 (statistically significant)     │
│   Edge weight: 0.68                              │
└──────────────────────────────────────────────────┘
```

**Output**: `pattern_signal = BEARISH`, `confidence = 68%`, `timeframe = 4h`

---

### Layer 5: Strategic Actor Analysis

**Input**: Microstructure variables (OB_imbalance, CVD, volume)

**Processing**: Model actor incentives and detect manipulation

```
Actor detection:

🐋 WHALE ANALYSIS:
├─ Order book imbalance: -0.15 (large sell walls at $65,100-65,200)
├─ CVD: -450 BTC (cumulative net selling)
├─ Volume: 3.4x average (high conviction)
└─ Conclusion: BEARISH
    Direction: Whale distribution
    Intensity: 0.7
    Reasoning: "Large sell walls + high CVD = institutional distribution"

🏦 MARKET MAKER ANALYSIS:
├─ Order book spread: 0.02% (tight)
├─ Book balance: Heavily imbalanced sell-side
└─ Conclusion: NEUTRAL
    Direction: Not resisting the move
    Intensity: 0.5
    Reasoning: "MMs allowing downside, no support placed"

🐟 RETAIL ANALYSIS:
├─ Funding rate: +0.02% → -0.01% (flipped from long bias)
├─ Liquidations: $12M longs liquidated
├─ Open interest: -8% (forced deleveraging)
└─ Conclusion: BULLISH (contrarian signal)
    Direction: Retail overleveraged long
    Intensity: 0.8
    Reasoning: "Retail long squeeze = contrarian bearish"

TRAP DETECTION:
├─ Pattern: "Engineered Long Squeeze"
├─ Match score: 72% (high)
├─ Severity: 0.65
├─ Risk level: HIGH
└─ Description: "Overleveraged longs liquidated by deliberate dump"
```

**Output**: `strategic_direction = BEARISH`, `confidence = 70%`, `trap_detected = "long_squeeze"`

---

### Layer 6: Swarm Simulation (500 Agents)

**Input**: Market state (RSI, MACD, EMA, volume)

**Processing**: 500 heterogeneous agents trade in simulation (Numba JIT)

```
Agent initialization (market-informed):
├─ Trend Followers (40%): 200 agents
│   └─ Initialize: EMA_9 < EMA_21 → bearish bias
├─ Mean Reverters (30%): 150 agents
│   └─ Initialize: RSI=58 (not oversold) → neutral bias
├─ Noise Traders (20%): 100 agents
│   └─ Random actions (market friction)
├─ Informed Traders (8%): 40 agents
│   └─ Use RSI + MACD signals → bearish bias
└─ Whales (2%): 10 agents
    └─ Large position sizes → selling pressure

Simulation: 100 time steps forward
for step in range(100):
    for agent in agents:
        action = agent.decide(market_state)
        execute_trade(agent, action)
        update_market_state()

Final positions after 100 steps:
├─ Bullish agents: 175 (35%)
├─ Bearish agents: 325 (65%)
├─ Neutral agents: 0 (0%)
└─ Consensus price trajectory: -4.2% average
```

**Output**: `swarm_direction = BEARISH`, `confidence = 65%`, `price_impact = -4.2%`

---

### Layer 7: Multi-Agent Hierarchy (62 Agents)

**Input**: 47 variable values from Layer 3

**Processing**: Hierarchical voting across 3 levels

```
LEVEL 1: Atomic Agents (47 agents, one per variable)
Each agent analyzes its variable and votes:
├─ RSI_14 Agent: "RSI=58, weakening momentum → BEARISH (0.6)"
├─ BB_WIDTH Agent: "BB expanding, volatility up → BEARISH (0.5)"
├─ OB_IMBALANCE Agent: "Heavy sell pressure → BEARISH (0.8)"
├─ CVD Agent: "Net selling -450 BTC → BEARISH (0.7)"
├─ ... (43 more agents)

Vote tally:
├─ Bearish: 34 agents (72%)
├─ Bullish: 8 agents (17%)
└─ Neutral: 5 agents (11%)

LEVEL 2: Group Leaders (10 agents, by category)
Aggregate votes from atomic agents:
├─ Momentum Group Leader: BEARISH (6/8 atomic agents bearish)
├─ Volatility Group Leader: BEARISH (7/9 atomic agents bearish)
├─ Volume Group Leader: BEARISH (3/4 atomic agents bearish)
├─ Microstructure Group Leader: BEARISH (5/6 atomic agents bearish)
├─ Trend Group Leader: BEARISH (6/9 atomic agents bearish)
└─ ... (5 more groups)

LEVEL 3: Super Groups (5 agents, by domain)
Aggregate votes from group leaders:
├─ ECONOMIC Super Group: BEARISH (confidence 0.75)
│   └─ Aggregates: Momentum, Volatility, Trend groups
├─ BEHAVIORAL Super Group: BEARISH (confidence 0.68)
│   └─ Aggregates: Volume, Market State groups
├─ PHYSICAL Super Group: BEARISH (confidence 0.72)
│   └─ Aggregates: Microstructure, Leverage groups
├─ SOCIAL Super Group: NEUTRAL (confidence 0.1)
│   └─ Aggregates: News sentiment (low significance)
└─ TEMPORAL Super Group: NEUTRAL (confidence 0.3)
    └─ Aggregates: Time-of-day, cycle position (no pattern)

Final hierarchy output: BEARISH (4/5 super groups, 1 neutral)
```

**Output**: `hierarchy_direction = BEARISH`, `confidence = 72%`

---

### Layer 8: Reflector (Meta-Reasoning)

**Input**: All layer outputs + decision logic

**Processing**: Examine reasoning for contradictions and blind spots

```
Meta-analysis:

✓ CONSISTENCY CHECK:
├─ Technical signals: Aligned (RSI, BB, volume all bearish)
├─ Microstructure: Aligned (OB imbalance, CVD bearish)
├─ Strategic: Aligned (whale distribution detected)
├─ Swarm: Aligned (65% bearish consensus)
└─ Result: No contradictions detected ✓

✓ CONFIDENCE ASSESSMENT:
├─ Multiple independent confirmations: ✓
├─ High-quality data sources: ✓
├─ Pattern has strong historical support (68%, n=127): ✓
└─ No contradictory signals: ✓

⚠ BLIND SPOT ANALYSIS:
├─ News sentiment is neutral (no fundamental driver)
│   Impact: Could be noise/volatility, not true trend change
├─ Funding rate flip is small (-0.01%, not extreme)
│   Impact: Not a strong contrarian signal yet
└─ Pattern is 8 months old (last reinforced 2024-03-15)
    Impact: May be stale, watch for regime change

Reflector conclusion: HIGH CONFIDENCE
Warnings: ["no_fundamental_driver", "watch_reversal", "pattern_age"]
```

**Output**: `reflection = HIGH_CONFIDENCE`, `warnings = ["no_fundamental", "watch_reversal"]`

---

### Layer 9: Narrative Sentinel (News Sentiment)

**Input**: News scan results from Layer 2 + price movement context

**Processing**: Correlate news with price action

```
News-Price Correlation Analysis:

News scan results:
├─ Reuters: No major BTC headlines
├─ Bloomberg: General market coverage
├─ CoinDesk: Technical analysis article (non-causal)
└─ Significance: 0.1 (very low)

Sentiment analysis:
├─ Keywords extracted: "volatility", "correction", "technical"
├─ Tone: Neutral to slightly bearish
├─ Emotional valence: -0.15 (mildly negative)
└─ Impact estimate: Minimal (0.1/1.0)

Causality check:
├─ News → Price correlation: 0.03 (very weak)
├─ Timeline: No significant news BEFORE price drop
└─ Conclusion: Price action is technical, not narrative-driven

Narrative assessment: NEUTRAL
```

**Output**: `news_sentiment = NEUTRAL`, `impact = 0.1`

---

### Layer 10: Fusion Gate (THE DECISION)

**Input**: Signals from all validation layers

**Processing**: Weighted vote aggregation with threshold enforcement

```
LAYER INPUTS:
┌─────────────┬───────────┬────────────┬────────┐
│ Layer       │ Signal    │ Confidence │ Weight │
├─────────────┼───────────┼────────────┼────────┤
│ Layer 4     │ BEARISH   │ 68%        │ 1.0    │
│ (Causal)    │           │            │        │
├─────────────┼───────────┼────────────┼────────┤
│ Layer 5     │ BEARISH   │ 70%        │ 1.0    │
│ (Strategic) │           │            │        │
├─────────────┼───────────┼────────────┼────────┤
│ Layer 6     │ BEARISH   │ 65%        │ 1.0    │
│ (Swarm)     │           │            │        │
├─────────────┼───────────┼────────────┼────────┤
│ Layer 7     │ BEARISH   │ 72%        │ 1.0    │
│ (Hierarchy) │           │            │        │
├─────────────┼───────────┼────────────┼────────┤
│ Layer 8     │ HIGH_CONF │ —          │ 0.5    │
│ (Reflector) │           │            │        │
├─────────────┼───────────┼────────────┼────────┤
│ Layer 9     │ NEUTRAL   │ 10%        │ 0.5    │
│ (News)      │           │            │        │
└─────────────┴───────────┴────────────┴────────┘

FUSION LOGIC:

Step 1: Count layer agreement
technical_layers = [Layer4, Layer5, Layer6, Layer7]
bearish_count = 4
consensus = 4/4 = 100%

Step 2: Calculate weighted average confidence
confidences = [68, 70, 65, 72]
avg_confidence = (68 + 70 + 65 + 72) / 4 = 68.75%

Step 3: Check for contradictions
if Layer9.significance > 0.5 AND Layer9.direction != bearish:
    avg_confidence *= 0.7  # Reduce confidence
    # NOT TRIGGERED (news significance = 0.1)

Step 4: Apply thresholds
if consensus >= 0.75 AND avg_confidence >= 60:
    signal = VALID
    if avg_confidence >= 80:
        action = EXECUTE
    else:
        action = OBSERVE_ONLY
```

**Decision output:**

```
┌────────────────────────────────────────────────────────────┐
│ FUSION GATE DECISION                                       │
├────────────────────────────────────────────────────────────┤
│ Direction: BEARISH                                         │
│ Confidence: 68.75%                                         │
│ Consensus: 100% (4/4 technical layers agree)              │
│                                                            │
│ Layer breakdown:                                           │
│   ├─ Causal Patterns: BEARISH (68%)                       │
│   ├─ Strategic Actors: BEARISH (70%)                      │
│   ├─ Swarm Simulation: BEARISH (65%)                      │
│   └─ Agent Hierarchy: BEARISH (72%)                       │
│                                                            │
│ Meta-checks:                                               │
│   ├─ Reflector: High confidence ✓                         │
│   ├─ News: Neutral (no contradiction) ✓                   │
│   └─ Contradictions: None detected                        │
│                                                            │
│ DECISION: SIGNAL VALID                                    │
│ ACTION: OBSERVE_ONLY                                      │
│                                                            │
│ Reasoning: Strong bearish consensus across all technical  │
│ layers, but confidence 68.75% < 80% execution threshold.  │
│ Log signal for evolution tracking. No trade executed.     │
└────────────────────────────────────────────────────────────┘
```

**Output**: Prediction logged to PostgreSQL for outcome verification

---

### Layer 11: Evolution Engine (4 Hours Later)

**Input**: Logged prediction + actual market outcome

**Processing**: Verify outcome, update agent performance

```
OUTCOME VERIFICATION:
Prediction made: 2024-11-15 14:32:00 UTC
├─ Direction: BEARISH
├─ Confidence: 68.75%
├─ Expected move: -2% to -5% within 4 hours
└─ Contributing agents: [47 agent IDs logged]

Wait 4 hours...

Outcome check: 2024-11-15 18:32:00 UTC
├─ Actual price: $62,650 (from $65,123)
├─ Actual move: -3.8%
├─ Direction accuracy: CORRECT ✓
├─ Magnitude accuracy: CORRECT ✓ (within -2% to -5% range)
└─ Timeframe accuracy: CORRECT ✓ (within 4 hours)

AGENT PERFORMANCE UPDATES:

FOR EACH of 34 bearish-voting agents:
    agent.correct_predictions += 1
    agent.win_rate = correct_predictions / total_predictions
    agent.performance_edge = win_rate - 0.5
    agent.consecutive_failures = 0  # Reset
    
Example: RSI_14 Agent
├─ Predictions: 248 (was 247)
├─ Correct: 169 (was 168)
├─ Win rate: 68.1% (was 68.0%)
├─ Performance edge: +18.1% (was +18.0%)
└─ Status: ACTIVE, Weight: 0.85

FOR EACH of 8 bullish-voting agents:
    agent.incorrect_predictions += 1
    agent.win_rate = correct_predictions / total_predictions
    agent.performance_edge = win_rate - 0.5
    agent.consecutive_failures += 1
    
    # Automatic evolution actions
    if agent.consecutive_failures >= 5:
        if agent.performance_edge < -0.20:
            agent.status = INVERTED  # Use contrarian signal
        elif agent.performance_edge < -0.10:
            agent.status = DISABLED  # Stop voting

Example: STOCH_RSI Agent (was wrong)
├─ Predictions: 216 (was 215)
├─ Correct: 60 (unchanged)
├─ Win rate: 27.8% (was 27.9%)
├─ Performance edge: -22.2% (was -22.1%)
├─ Consecutive failures: 6 (was 5)
└─ Status: INVERTED (edge < -20%)

CAUSAL PATTERN UPDATES:
pattern["RSI_drop_OB_bearish"]:
├─ Accuracy: 68.2% (was 68.0%, +0.2%)
├─ Sample count: 128 (was 127)
├─ Last reinforced: 2024-11-15 (was 2024-03-15)
└─ Edge weight: 0.682 (updated in DAG)
```

**Output**: Updated agent performance metrics in PostgreSQL, reinforced causal patterns

---

### Layer 12: Execution & Position Management

**Input**: Fusion gate decision

**Processing**: Risk management and order routing

```
Decision: OBSERVE_ONLY (confidence 68.75% < 80%)
├─ No trade executed
├─ Signal logged for tracking
└─ Evolution data collected

If confidence ≥ 80%, would have executed:
├─ Position sizing: Kelly Criterion
│   └─ f* = (p*b - q) / b where p=0.80, b=1.5, q=0.20
│   └─ Optimal fraction: 0.47 (47% of capital)
├─ Stop loss: -2% (risk management)
├─ Take profit: +4% (2:1 reward:risk)
└─ Order type: Limit order at $64,900
```

**Output**: No execution (observation mode)

---

### Layer 13: Memory & Knowledge Accumulation

**Input**: Prediction + outcome + agent performance

**Processing**: Persist to multiple storage systems

```
PostgreSQL (structured data):
├─ INSERT INTO predictions (...) VALUES (...)
├─ UPDATE agent_evolution SET win_rate = ..., performance_edge = ...
└─ UPDATE causal_patterns SET accuracy = ..., sample_count = ...

Qdrant (vector embeddings):
└─ Store prediction embedding for similarity search

Wiki (markdown knowledge base):
└─ Write to wiki/lessons/BTC_USDT/prediction_2024-11-15.md:
    """
    **Prediction:** BEARISH 68.75%
    **Outcome:** CORRECT (-3.8% in 4h)
    **Pattern:** RSI_drop_OB_bearish (n=128, 68.2% accuracy)
    **Key lesson:** OB_imbalance < -0.15 + volume > 2x = reliable signal
    **Agents:** RSI_14 (68%), OB_IMBALANCE (80%), CVD (70%)
    **Blind spots:** No fundamental driver, watch for reversal
    """
```

**Output**: Historical knowledge base updated

---

### Layer 14: Error Recovery & Fallback

**Processing**: Resilience mechanisms (always active)

```
LLM Failover (for optional LLM debate):
├─ Primary: Anthropic API
├─ Backup: AWS Bedrock
└─ Emergency: Local Ollama

Database Failover:
├─ Connection pool exhaustion → reconnect with backoff
├─ Query timeout → retry with degraded query
└─ PostgreSQL down → use Redis cache

WebSocket Recovery:
├─ Connection lost → reconnect with exponential backoff
├─ Data gap → backfill from REST API
└─ Invalid data → skip frame, log error

Empirical Fallbacks:
├─ Variable computation fails → use cached values
├─ Layer timeout → use last known signal
└─ All layers fail → output "INSUFFICIENT_DATA"
```

**Output**: System remains operational despite failures

---

## The Complete 14-Layer Architecture

Visual representation of data flow:

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT: Raw Market Data                                             │
│  ├─ Binance WebSocket: trades, order books, liquidations           │
│  ├─ Yahoo Finance: OHLCV, volume                                    │
│  └─ RSS Feeds: news, sentiment                                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1: Sensory Data Ingestion                                    │
│  Purpose: Capture and timestamp all incoming data                   │
│  Output: Raw data in TimescaleDB                                    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 2: News Intelligence Scanner                                 │
│  Purpose: Detect market-moving events from news                     │
│  Method: CRUCIX OSINT (27 sources)                                  │
│  Output: news_sentiment, significance_score                         │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 3: Variable Computation Engine                               │
│  Purpose: Decompose market state into 47 atomic measurements        │
│  Method: Polars vectorized operations (10-50x faster than pandas)   │
│  Output: 47 timestamped variable values                             │
│    ├─ Trend (9): EMA, VWAP, slopes                                  │
│    ├─ Momentum (8): RSI, MACD, Stochastic                           │
│    ├─ Volatility (9): Bollinger, ATR, GARCH                         │
│    ├─ Volume (4): OBV, relative volume                              │
│    ├─ Microstructure (6): Order book imbalance, CVD                 │
│    ├─ Leverage (3): Funding rates, OI                               │
│    ├─ Temporal (3): Time-of-day, day-of-week                        │
│    └─ Market State (5): Price, regime                               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 4: Causal Pattern Discovery                                  │
│  Purpose: Find structural relationships and historical patterns     │
│  Method: NOTEARS/PC/Granger DAG + empirical pattern tracking        │
│  Output: pattern_signal, confidence, timeframe                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 5: Strategic Actor Analysis                                  │
│  Purpose: Model actor incentives and detect manipulation            │
│  Actors: Whales, Market Makers, Retail, Miners, Institutions        │
│  Output: strategic_direction, confidence, trap_detected             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 6: Swarm Simulation (500 Agents)                             │
│  Purpose: Simulate market microstructure                            │
│  Method: Numba JIT-compiled simulation (100 steps forward)          │
│  Agents: Trend (40%), Mean Revert (30%), Noise (20%), etc.          │
│  Output: swarm_direction, confidence, price_impact                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 7: Multi-Agent Hierarchy (62 Agents)                         │
│  Purpose: Hierarchical voting system for consensus                  │
│  Structure:                                                          │
│    Level 1: 47 atomic agents (one per variable)                     │
│    Level 2: 10 group leaders (by category)                          │
│    Level 3: 5 super groups (by domain)                              │
│  Output: hierarchy_direction, confidence                            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 8: Reflector (Meta-Reasoning)                                │
│  Purpose: Examine reasoning for contradictions                      │
│  Checks: Consistency, confidence, blind spots                       │
│  Output: reflection_status, warnings                                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 9: Narrative Sentinel (News Sentiment)                       │
│  Purpose: Correlate news with price movements                       │
│  Sources: Reuters, Bloomberg, CoinDesk, RSS feeds                   │
│  Output: news_sentiment, impact_score                               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 10: Fusion Gate (DECISION)                                   │
│  Purpose: Combine all layer outputs into final decision             │
│  Logic:                                                              │
│    1. Count layer agreement (consensus threshold: 75%)              │
│    2. Calculate weighted average confidence                         │
│    3. Check for contradictions (news vs technical)                  │
│    4. Apply thresholds: >60% signal, >80% execute                   │
│  Output: final_direction, confidence, action                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 11: Evolution Engine (Learning Loop)                         │
│  Purpose: Track performance and auto-evolve system                  │
│  Process:                                                            │
│    1. Log prediction with contributing agents                       │
│    2. Wait for outcome (4 hours)                                    │
│    3. Update agent win_rates and performance_edge                   │
│    4. Auto-prune: edge < -10% → disabled                            │
│    5. Auto-invert: edge < -20% → use contrarian signal              │
│  Output: Updated agent performance in PostgreSQL                    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 12: Execution & Position Management                          │
│  Purpose: Risk management and order execution                       │
│  Logic: Kelly Criterion position sizing, stop losses                │
│  Output: Order sent to exchange (if confidence ≥80%)                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 13: Memory & Knowledge Accumulation                          │
│  Purpose: Store patterns, agent performance, historical context     │
│  Storage: PostgreSQL + TimescaleDB + Qdrant + Wiki                  │
│  Output: Historical knowledge base updated                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 14: Error Recovery & Fallback                                │
│  Purpose: Graceful degradation on failures                          │
│  Mechanisms:                                                         │
│    ├─ LLM failover: Anthropic → Bedrock → Ollama                   │
│    ├─ Database failover: Reconnect with backoff                     │
│    ├─ WebSocket recovery: Reconnect + backfill                      │
│    └─ Empirical fallbacks: Cached values, last signals              │
│  Output: System stays operational despite failures                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  OUTPUT: Final Decision + Action                                    │
│  ├─ Direction: BEARISH                                              │
│  ├─ Confidence: 68.75%                                              │
│  ├─ Action: OBSERVE_ONLY                                            │
│  └─ Logged: For outcome verification and learning                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Causal Discovery vs Empirical Correlation

**Causal discovery algorithms (Layer 4):**

Run three structural learning algorithms on variable time-series:

- **NOTEARS**: Continuous optimization for directed acyclic graph discovery
- **PC Algorithm**: Constraint-based causal structure learning via conditional independence
- **Granger Causality**: Time-series precedence testing (does X's past predict Y's future?)

**Output**: Directed acyclic graph where edges represent discovered causal relationships. Edges confirmed by 2+ algorithms receive higher weight.

**Limitation**: Observational causality only. True causal inference requires intervention (randomized controlled trials), which is impossible in financial markets. These algorithms discover *predictive structure* and *temporal precedence*, not scientific causation.

**Empirical pattern tracking (Layer 4 + Layer 11):**

Historical pattern recognition with continuous accuracy monitoring:

```
Pattern example:
IF RSI > 70 initially AND RSI drops >10 points in 15min 
AND order_book_imbalance < -0.10 AND volume > 2x average
THEN price continues dropping within 4h

Historical accuracy: 68.2% (n=128 samples)
Last reinforced: 2024-11-15
Pattern weight: 0.682
```

Every prediction is verified against actual outcomes 4 hours later. Patterns below 60% accuracy are pruned. Recent outcomes weighted more heavily than old ones.

**Combined approach**: Causal DAG provides structural hypotheses, empirical patterns provide predictive signals. Fusion gate validates both must agree before generating predictions.

---

## Self-Evolution: Three-Level Learning

### Level 1: Agent Evolution (Per-Prediction)

Every prediction logs which agents voted bearish/bullish/neutral. Four hours later, outcome is verified and agent performance updated.

```python
# src/atomicx/agents/base.py
class BaseAgent:
    def update_performance(self, correct: bool):
        self.win_rate = correct_predictions / total_predictions
        self.performance_edge = win_rate - 0.5  # Edge over coin flip
        
        # Automatic actions
        if total_predictions >= 200:
            if performance_edge < -0.10:
                self.status = "DISABLED"  # Stop voting
            elif performance_edge < -0.20:
                self.strategy = "INVERTED"  # Use contrarian signal
```

**Example agent trajectories:**

```
RSI_14 Agent:
├─ Predictions: 247 | Win rate: 68% | Edge: +18%
├─ Status: ACTIVE | Weight: 0.85 (increased from 0.65)
└─ Action: Continue normally

STOCH_RSI Agent:
├─ Predictions: 215 | Win rate: 28% | Edge: -22%
├─ Status: INVERTED | Weight: 0.20 (low trust)
└─ Action: Use contrarian signal (invert vote)

BB_WIDTH Agent:
├─ Predictions: 203 | Win rate: 42% | Edge: -8%
├─ Status: ACTIVE (watching) | Weight: 0.35 (decreased)
└─ Action: Continue but reduced influence
```

### Level 2: Strategy Evolution (Daily Cycle)

```python
# src/atomicx/memory/evolution.py
class DailyEvolutionCycle:
    def run_evolution(self):
        # 1. Retire: edge decay >70%
        # 2. Mutate: top 20% performers (±15% parameter variance)
        # 3. Spawn: new strategies if active population < 3
```

**Evolution cycle example:**
```
CYCLE #47:
├─ Retired: "momentum_slow_v12" (edge decay 72%)
├─ Mutated: "volatility_fast_v8" → "volatility_fast_v8_mutant_v47"
│   └─ BB_period: 20→18 (-10%), BB_std: 2.0→2.3 (+15%), ATR_period: 14→16 (+14%)
├─ Spawned: "auto_spawn_47" (genome thin)
└─ Genome: 8 active, 3 retired, 2 mutants
```

Mutants compete with parents. If mutant outperforms parent after 50 predictions, parent is retired.

### Level 3: Architectural Evolution (Meta-Reasoning)

```python
# src/atomicx/brain/evolver.py
class EvolverAgent:
    async def analyze_and_propose(self):
        # Analyze last 20 decision monologues
        # Propose system-level changes
        # Examples:
        # - "Decrease swarm layer trust by 20%" (if low regime alignment)
        # - "Add tiebreaker agent for high-conflict scenarios"
        # - "Increase fusion gate threshold to 75%" (if false positive rate high)
```

Proposals require manual approval before enacting architectural changes.

---

## The Journey: Raw Data → Decision

**Concrete example: BTC drops 3% in 15 minutes ($67,234 → $65,123)**

### Layer 1: Sensory Ingestion
```
Binance WebSocket → Raw data
├─ Trades: 2.3M BTC volume in 15min
├─ Order book: 450 BTC sell orders at $65,100-65,200
├─ Order book: 220 BTC buy orders at $64,800-65,000
├─ Liquidations: $12M long positions liquidated
└─ CVD: -450 BTC net selling
```

### Layer 3: Variable Decomposition
```
47 variables computed (showing 7):
├─ momentum_rsi_14: 72 → 58 (momentum weakening)
├─ volatility_bb_bandwidth: 0.008 → 0.012 (volatility expanding)
├─ microstructure_ob_imbalance: -0.05 → -0.15 (strong sell pressure)
├─ volume_rel_volume: 1.2x → 3.4x (high activity)
├─ microstructure_cvd: -120 → -450 (net selling acceleration)
├─ trend_ema_9: $66,800 → $65,400 (downtrend)
└─ leverage_funding_rate: +0.02% → -0.01% (longs → shorts)
```

### Layer 4: Causal Pattern Discovery
```
DAG search finds matching pattern:
┌──────────────────────────────────────────────────┐
│ Pattern: RSI_drop_OB_bearish                     │
├──────────────────────────────────────────────────┤
│ IF RSI > 70 initially                            │
│ AND RSI drops >10 points in 15min               │
│ AND OB_imbalance < -0.10                         │
│ AND volume > 2x average                          │
│ THEN continued drop within 4h                    │
│                                                  │
│ Historical accuracy: 68% (n=127 samples)         │
│ Last reinforced: 2024-03-15 (8 months ago)       │
└──────────────────────────────────────────────────┘

Output: pattern_signal=BEARISH, confidence=68%, timeframe=4h
```

### Layer 5: Strategic Actor Analysis
```
Actor detection:
├─ 🐋 WHALE: Direction=BEARISH, Intensity=0.7
│   └─ Evidence: OB_imbalance=-0.15, CVD=-450, volume=3.4x
│   └─ Reasoning: "Large sell walls + high CVD = whale distribution"
├─ 🏦 MARKET MAKER: Direction=NEUTRAL, Intensity=0.5
│   └─ Evidence: Tight spread, imbalanced book
│   └─ Reasoning: "Not resisting the move"
└─ 🐟 RETAIL: Direction=BULLISH, Intensity=0.8 (contrarian)
    └─ Evidence: Funding +0.02% → -0.01%, $12M longs liquidated
    └─ Reasoning: "Retail long squeeze = contrarian bearish"

Trap detection: "Engineered Long Squeeze" (72% match, severity 0.65)

Output: strategic_direction=BEARISH, confidence=70%
```

### Layer 6: Swarm Simulation
```
500 agents initialized based on current state:
├─ Trend followers (40%): Bearish bias (EMA_9 < EMA_21)
├─ Mean reverters (30%): Neutral bias (RSI=58, not oversold)
├─ Noise traders (20%): Random actions
├─ Informed traders (8%): Bearish bias (RSI + MACD signals)
└─ Whales (2%): Selling pressure

Simulation: 100 time steps forward (Numba JIT)
Result: 325 bearish (65%), 175 bullish (35%), average price -4.2%

Output: swarm_direction=BEARISH, confidence=65%
```

### Layer 7: Agent Hierarchy
```
Level 1 (47 atomic agents):
├─ Bearish: 34 agents (72%)
├─ Bullish: 8 agents (17%)
└─ Neutral: 5 agents (11%)

Level 2 (10 group leaders):
├─ Momentum Group: BEARISH (6/8 atomic bearish)
├─ Volatility Group: BEARISH (7/9 atomic bearish)
├─ Microstructure Group: BEARISH (5/6 atomic bearish)
└─ ... 7 more groups

Level 3 (5 super groups):
├─ ECONOMIC Super: BEARISH (confidence 0.75)
├─ BEHAVIORAL Super: BEARISH (confidence 0.68)
├─ PHYSICAL Super: BEARISH (confidence 0.72)
├─ SOCIAL Super: NEUTRAL (news neutral)
└─ TEMPORAL Super: NEUTRAL (no time pattern)

Output: hierarchy_direction=BEARISH, confidence=72%
```

### Layer 8: Meta-Reasoning
```
Reflector analysis:
✓ Consistency: Technical, microstructure, strategic all aligned
✓ Confidence: Multiple independent confirmations
✓ Pattern support: Strong historical backing (68%, n=127)
⚠ Blind spot: No fundamental driver (news neutral)
⚠ Watch: Funding rate flip is small (-0.01%, not extreme)

Output: reflection=HIGH_CONFIDENCE, warnings=["no_fundamental", "watch_reversal"]
```

### Layer 9: Narrative Analysis
```
News scan: Reuters, Bloomberg, CoinDesk
├─ Keywords: "volatility", "correction", "technical"
├─ Significance: 0.1 (very low)
├─ Tone: Neutral to slightly bearish
└─ Conclusion: Price action is technical, not narrative-driven

Output: news_sentiment=NEUTRAL, impact=0.1
```

### Layer 10: Fusion Gate (THE DECISION)
```
Layer inputs:
├─ Layer 4 (Causal): BEARISH 68%
├─ Layer 5 (Strategic): BEARISH 70%
├─ Layer 6 (Swarm): BEARISH 65%
├─ Layer 7 (Hierarchy): BEARISH 72%
├─ Layer 8 (Reflector): HIGH_CONFIDENCE
└─ Layer 9 (News): NEUTRAL 10%

Fusion logic:
├─ Technical layer consensus: 4/4 (100%)
├─ Weighted average confidence: 68.75%
├─ News contradiction check: PASS (neutral, no conflict)
└─ Threshold check: 100% consensus ✓, 68.75% < 80% threshold

┌────────────────────────────────────────────────┐
│ DECISION                                       │
├────────────────────────────────────────────────┤
│ Direction: BEARISH                             │
│ Confidence: 68.75%                             │
│ Consensus: 100% (4/4 technical layers)        │
│                                                │
│ ACTION: SIGNAL VALID, OBSERVE_ONLY            │
│                                                │
│ Reasoning: Strong bearish consensus but below │
│ 80% execution threshold. Log for tracking.    │
└────────────────────────────────────────────────┘
```

### Layer 11: Evolution (4 Hours Later)
```
Outcome verification:
├─ Actual: BTC -3.8% ($65,123 → $62,650)
├─ Prediction: CORRECT ✓
└─ Timeframe: Within 4 hours ✓

Agent performance updates:
FOR EACH of 34 bearish-voting agents:
    correct_predictions++, win_rate++, consecutive_failures=0
FOR EACH of 8 bullish-voting agents:
    incorrect_predictions++, win_rate--, consecutive_failures++
    IF consecutive_failures >= 5 AND edge < -0.10:
        status = DISABLED or INVERTED

Causal pattern updates:
pattern["RSI_drop_OB_bearish"].accuracy = 68.2% (was 68.0%)
pattern.sample_count = 128 (was 127)
pattern.last_reinforced = 2024-11-15
```

---

## Variable Architecture: Domain-Agnostic Design

Variables are not hardcoded. Each variable is defined via universal schema:

```python
from atomicx.variables.types import VariableDefinition, VariableDomain

VariableDefinition(
    id="variable_identifier",           # Domain-specific
    name="Human Readable Name",
    domain=VariableDomain.ECONOMIC,      # Universal domain
    category="sub_category",             # Flexible
    source="data_source",
    update_frequency="1m",
    causal_half_life=24.0,              # Signal decay (hours)
    reliability_score=0.8,               # Data quality
    params={"period": 14}                # Computation params
)
```

**Seven universal domains:**
- **PHYSICAL**: Tangible measurements (order flow, temperature, inventory)
- **BIOLOGICAL**: Growth/decay patterns (network effects, viral spread)
- **PSYCHOLOGICAL**: Sentiment, emotions, narratives
- **BEHAVIORAL**: Observed actions (trading patterns, buying behavior)
- **ECONOMIC**: Resource allocation (price, demand, supply)
- **SOCIAL**: Community signals (news, social media)
- **TEMPORAL**: Time-based patterns (seasonality, cycles)

**Current implementation: 47 financial variables**
- Trend (9): EMA, VWAP, slopes
- Momentum (8): RSI, MACD, Stochastic RSI, ROC
- Volatility (9): Bollinger Bands, ATR, GARCH, Keltner
- Volume (4): OBV, relative volume, VPT
- Microstructure (6): Order book imbalance, CVD, bid/ask volumes
- Leverage (3): Funding rates, open interest
- Temporal (3): Time-of-day, day-of-week, cycle position
- Market State (5): Price, regime, regime shift probability

**Domain expansion (design, not validated):**

Weather prediction:
```python
VariableDefinition(id="temperature_surface", domain=PHYSICAL, category="thermal", ...)
VariableDefinition(id="pressure_barometric", domain=PHYSICAL, category="atmospheric", ...)
```

Medical diagnosis:
```python
VariableDefinition(id="vital_heart_rate", domain=BIOLOGICAL, category="cardiovascular", ...)
VariableDefinition(id="biomarker_wbc_count", domain=BIOLOGICAL, category="hematology", ...)
```

See `docs/example_domains.py` for complete examples.

---

## Validation Requirements

**Current status**: Operational on live financial data. No published performance metrics.

**Required for production:**

1. **Out-of-sample validation**
   - Train on 2020-2022 data (pattern discovery)
   - Validate on 2023 data (threshold tuning)
   - Test on 2024 data (never-seen accuracy)

2. **Baseline comparison**
   - Buy-and-hold BTC/USDT
   - Moving average crossover (SMA 50/200)
   - Logistic regression (RSI + volume)
   - Random forest baseline

3. **Ablation studies**
   - All 14 layers vs 7 layers vs 1 layer
   - Contribution of each layer to accuracy
   - Justification for architectural complexity

4. **Calibration analysis**
   - Predicted confidence vs actual accuracy
   - Plot calibration curve (predicted probability vs outcome)
   - Brier score calculation

5. **Walk-forward validation**
   - Re-discover patterns every 3 months using only past data
   - Test on next 3 months (strict temporal separation)
   - Report degradation over time

---

## Known Limitations

1. **Domain specificity**: Validated exclusively on financial markets
2. **Causality scope**: Observational causal discovery, not interventional (no RCTs possible in markets)
3. **Benchmark absence**: No published comparisons vs simple baselines
4. **Computational cost**: Swarm simulation CPU-intensive (mitigated by Numba JIT)
5. **Data latency**: WebSocket ~60-200ms delay
6. **Incomplete data**: Funding rate not currently collected (degrades strategic layer)
7. **Research stage**: Not production-ready for capital deployment

---

## Academic & Industry Foundations

**Causal Inference:**
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- DoWhy (Microsoft Research), EconML (Microsoft Research), causal-learn (CMU), gCastle (Alibaba)

**Agent-Based Modeling:**
- Tesfatsion, L. (2006). *Agent-Based Computational Economics*. Handbook of Computational Economics, 2, 831-880.
- Santa Fe Institute: Complexity economics research

**Multi-Agent Systems:**
- Du, Y., et al. (2023). *Improving Factuality and Reasoning in Language Models through Multiagent Debate*. arXiv:2305.14325.
- Bai, Y., et al. (2022). *Constitutional AI: Harmlessness from AI Feedback*. arXiv:2212.08073.
- Yao, S., et al. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*. arXiv:2305.10601.

**Quantitative Finance:**
- Kelly Criterion (Ed Thorp): Optimal position sizing
- Dual-Confirmation (Renaissance Technologies): Multi-model validation
- Regime Detection (Two Sigma): Market state classification
- Anti-Learning (Citadel): Inverting wrong models

**Open Source Integrations:**
- [OpenClaw](https://github.com/openclaw/openclaw): LLM failover (`src/atomicx/brain/llm_profiles.py`), browser pool (`src/atomicx/intelligence/browser_pool.py`), lane queues (`src/atomicx/agents/lane_queue.py`)
- [Karpathy LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f): Knowledge base (`src/atomicx/memory/wiki.py`)
- [CRUCIX](https://github.com/calesthio/Crucix): OSINT intelligence terminal (27 sources: NASA FIRMS, OpenSky, ACLED, GDELT) — integrated as `src/crucix/`

**Memory Systems:**
- Packer, C., et al. (2023). *MemGPT: Towards LLMs as Operating Systems*. arXiv:2310.08560.

**Software Stack:**
- PostgreSQL, TimescaleDB, Redis, Kafka, Qdrant
- FastAPI, Uvicorn, SQLAlchemy, Alembic
- Polars, NumPy, SciPy, Numba
- PyTorch, scikit-learn, XGBoost, LightGBM
- Sentence-Transformers, Hugging Face Transformers
- LangGraph, LangChain, CCXT, yfinance
- Playwright, feedparser, httpx

---

## Installation

### Prerequisites
- Python 3.11+
- PostgreSQL 14+ with TimescaleDB extension
- 8GB+ RAM (16GB recommended)

### Quick Start
```bash
# Clone repository
git clone https://github.com/shaonsikder1952/atomicx.git
cd atomicx

# Install dependencies
pip install -e ".[all]"  # Full installation

# Database setup
brew install timescaledb  # macOS
psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"

# Configure
cp .env.example .env
# Edit .env: DATABASE_URL, ANTHROPIC_API_KEY (optional), BINANCE_API_KEY (optional)

# Migrate database
python run_migration.py

# Run system
python run.py

# Dashboard: http://localhost:8001
```

---

## Project Structure

```
atomicx/
├── src/atomicx/
│   ├── agents/              # 62-agent hierarchy (orchestrator.py, base.py)
│   ├── brain/               # Orchestrator, reflector, evolver, debate
│   ├── causal/              # NOTEARS, PC, Granger (engine.py)
│   ├── data/                # WebSocket/REST connectors (binance_ws.py)
│   ├── dashboard/           # FastAPI backend, HTML/CSS/JS frontend
│   ├── evolution/           # Performance tracking
│   ├── execution/           # Order management, position sizing
│   ├── fusion/              # Multi-layer consensus (engine.py)
│   ├── intelligence/        # News scanner, browser pool, CRUCIX integration
│   ├── memory/              # Wiki (wiki.py), evolution (evolution.py), Qdrant
│   ├── narrative/           # Sentiment analysis
│   ├── strategic/           # Actor analysis (whales, market makers)
│   ├── swarm/               # 500-agent Numba simulation
│   └── variables/           # 47 variable definitions (catalog.py, engine.py)
├── src/crucix/              # OSINT intelligence terminal (27 sources)
├── alembic/                 # Database migrations
├── config/                  # YAML configuration
├── docs/                    # Architecture documentation
└── run.py                   # System entry point
```

---

## Dashboard

**Access**: `http://localhost:8001`

**Pages:**
- `/` — Main intelligence terminal (real-time predictions)
- `/causality.html` — Decision audit viewer (14-layer breakdown)
- `/evolution.html` — Agent performance tracking
- `/god_mode_full.html` — Omniscient view (all variables, all agents)
- `/db.html`, `/mem.html`, `/diary.html` — Quick data views

**Technology**: FastAPI + Server-Sent Events (SSE) for real-time updates. No manual refresh required.

---

**Version**: 1.0  
**Status**: Research Stage  
**Last Updated**: 2026-04-06

---

## Disclaimer

This system is for research purposes. Does not constitute financial advice. Past performance does not guarantee future results. No claims of profitability. Use at own risk.
