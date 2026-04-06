"""
Examples of how AtomicX variables are domain-configurable.
The system doesn't know about "RSI" or "MACD" - those are just
the current financial domain configuration.
"""

from atomicx.variables.types import VariableDefinition, VariableDomain, VariableTimeframe

# ============================================================================
# DOMAIN 1: FINANCIAL MARKETS (current implementation)
# ============================================================================
financial_variables = [
    VariableDefinition(
        id="momentum_rsi_14",  # Generic: momentum type, rsi method, 14 period
        name="Relative Strength Index (14)",
        domain=VariableDomain.ECONOMIC,
        category="momentum",
        source="market_data",
        params={"period": 14},
    ),
    VariableDefinition(
        id="volatility_bb_bandwidth",
        name="Bollinger Band Width",
        domain=VariableDomain.ECONOMIC,
        category="volatility",
        source="market_data",
        params={"period": 20, "std_dev": 2.0},
    ),
]

# ============================================================================
# DOMAIN 2: WEATHER PREDICTION
# ============================================================================
weather_variables = [
    VariableDefinition(
        id="temperature_surface",
        name="Surface Temperature",
        domain=VariableDomain.PHYSICAL,
        category="thermal",
        source="noaa_weather_station",
        update_frequency=VariableTimeframe.M15,
        causal_half_life=6.0,  # Temperature persists ~6 hours
    ),
    VariableDefinition(
        id="pressure_barometric",
        name="Barometric Pressure",
        domain=VariableDomain.PHYSICAL,
        category="atmospheric",
        source="noaa_weather_station",
        update_frequency=VariableTimeframe.M15,
        causal_half_life=12.0,  # Pressure systems persist longer
    ),
    VariableDefinition(
        id="humidity_relative",
        name="Relative Humidity",
        domain=VariableDomain.PHYSICAL,
        category="moisture",
        source="noaa_weather_station",
        update_frequency=VariableTimeframe.M15,
        causal_half_life=4.0,
    ),
    VariableDefinition(
        id="wind_speed_surface",
        name="Surface Wind Speed",
        domain=VariableDomain.PHYSICAL,
        category="wind",
        source="noaa_weather_station",
        update_frequency=VariableTimeframe.M5,
        causal_half_life=2.0,  # Wind changes quickly
    ),
]

# ============================================================================
# DOMAIN 3: MEDICAL DIAGNOSIS
# ============================================================================
medical_variables = [
    VariableDefinition(
        id="vital_heart_rate",
        name="Heart Rate (BPM)",
        domain=VariableDomain.BIOLOGICAL,
        category="cardiovascular",
        source="patient_monitor",
        update_frequency=VariableTimeframe.M1,
        causal_half_life=0.5,  # Changes rapidly
    ),
    VariableDefinition(
        id="vital_blood_pressure_systolic",
        name="Systolic Blood Pressure",
        domain=VariableDomain.BIOLOGICAL,
        category="cardiovascular",
        source="patient_monitor",
        update_frequency=VariableTimeframe.M15,
        causal_half_life=2.0,
    ),
    VariableDefinition(
        id="biomarker_wbc_count",
        name="White Blood Cell Count",
        domain=VariableDomain.BIOLOGICAL,
        category="hematology",
        source="lab_results",
        update_frequency=VariableTimeframe.D1,  # Lab results daily
        causal_half_life=24.0,
    ),
    VariableDefinition(
        id="symptom_fever_severity",
        name="Fever Severity (self-reported)",
        domain=VariableDomain.PSYCHOLOGICAL,
        category="symptoms",
        source="patient_survey",
        update_frequency=VariableTimeframe.H4,
        reliability_score=0.6,  # Self-reported less reliable
    ),
]

# ============================================================================
# DOMAIN 4: SUPPLY CHAIN
# ============================================================================
supply_chain_variables = [
    VariableDefinition(
        id="inventory_warehouse_count",
        name="Warehouse Inventory Count",
        domain=VariableDomain.PHYSICAL,
        category="inventory",
        source="warehouse_management_system",
        update_frequency=VariableTimeframe.H1,
        causal_half_life=168.0,  # Inventory persists ~1 week
    ),
    VariableDefinition(
        id="demand_forecast_7day",
        name="7-Day Demand Forecast",
        domain=VariableDomain.ECONOMIC,
        category="demand",
        source="demand_planning_model",
        update_frequency=VariableTimeframe.D1,
        causal_half_life=72.0,
    ),
    VariableDefinition(
        id="logistics_shipping_delay_avg",
        name="Average Shipping Delay (hours)",
        domain=VariableDomain.PHYSICAL,
        category="logistics",
        source="shipping_tracking",
        update_frequency=VariableTimeframe.H4,
        causal_half_life=48.0,
    ),
    VariableDefinition(
        id="sentiment_supplier_reliability",
        name="Supplier Reliability Score",
        domain=VariableDomain.BEHAVIORAL,
        category="supplier_quality",
        source="supplier_performance_db",
        update_frequency=VariableTimeframe.W1,
        causal_half_life=720.0,  # Supplier reputation persists months
    ),
]

# ============================================================================
# THE PATTERN: Domain-Agnostic Variable Schema
# ============================================================================
# All use the same VariableDefinition schema:
#   - id: unique identifier (domain-specific naming)
#   - name: human-readable
#   - domain: PHYSICAL, BIOLOGICAL, PSYCHOLOGICAL, BEHAVIORAL, ECONOMIC, SOCIAL, TEMPORAL
#   - category: flexible sub-category
#   - source: where data comes from (API, sensor, survey, etc.)
#   - update_frequency: how often it updates
#   - causal_half_life: how long the signal persists
#   - reliability_score: data quality (0-1)
#
# The AtomicX architecture doesn't care WHAT the variables measure.
# It only cares about:
#   1. Can we measure it? (get_value())
#   2. Does it change over time? (timestamps)
#   3. Does it predict outcomes? (tracked via evolution engine)
