import numpy as np
import pandas as pd

INPUT_PATH = r"C:\Users\Admin\Desktop\wild-rift-churn-prediction\online_gaming_behavior_dataset.csv"
OUTPUT_PATH = r"C:\Users\Admin\Desktop\wild-rift-churn-prediction\Synthetic_wr_Dataset.csv"
RANDOM_STATE = 42
TARGET_RATE = 0.22

rng = np.random.default_rng(RANDOM_STATE)


def clip_round(arr, low, high):
    return np.clip(np.rint(arr), low, high).astype(int)


def weighted_choice(values, probs, size):
    return rng.choice(values, size=size, p=probs)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -----------------------------
# 1. Load source dataset
# -----------------------------
df = pd.read_csv(INPUT_PATH).copy()
n = len(df)

# Basic cleanup
for col in ["Age", "PlayTimeHours", "InGamePurchases", "SessionsPerWeek", "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill minimal missing values for seed use
for col in ["Age", "PlayTimeHours", "InGamePurchases", "SessionsPerWeek", "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

for col in ["Gender", "Location", "GameGenre", "GameDifficulty", "EngagementLevel"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# -----------------------------
# 2. Assign SEA region mix
# -----------------------------
regions = ["ID", "PH", "TH", "VN", "MY", "SG"]
region_probs = [0.24, 0.22, 0.18, 0.16, 0.12, 0.08]
region = weighted_choice(regions, region_probs, n)

# -----------------------------
# 3. Player lifecycle segments
# -----------------------------
segment = weighted_choice(["new", "midcore", "veteran"], [0.30, 0.50, 0.20], n)

account_age_days = np.zeros(n, dtype=int)
account_age_days[segment == "new"] = rng.integers(1, 31, size=(segment == "new").sum())
account_age_days[segment == "midcore"] = rng.integers(31, 181, size=(segment == "midcore").sum())
account_age_days[segment == "veteran"] = rng.integers(181, 366, size=(segment == "veteran").sum())

# -----------------------------
# 4. Device / role / queue
# -----------------------------
device_tier = weighted_choice(["low", "mid", "high"], [0.35, 0.45, 0.20], n)
preferred_role = weighted_choice(["baron", "jungle", "mid", "dragon", "support"], [0.18, 0.20, 0.24, 0.20, 0.18], n)
preferred_queue = weighted_choice(["normal_pvp", "ranked", "aram"], [0.45, 0.40, 0.15], n)

# -----------------------------
# 5. Build WR-like behavioral features from seed columns
# -----------------------------
engagement_map = {"Low": -1.0, "Medium": 0.0, "High": 1.0, "Unknown": 0.0}
engagement_score = df["EngagementLevel"].map(engagement_map).fillna(0.0).to_numpy()

difficulty_map = {"Easy": 0.2, "Medium": 0.5, "Hard": 0.8, "Unknown": 0.5}
difficulty_score = df["GameDifficulty"].map(difficulty_map).fillna(0.5).to_numpy()

base_sessions = df["SessionsPerWeek"].to_numpy()
base_playtime = df["PlayTimeHours"].to_numpy()
base_avg_session = df["AvgSessionDurationMinutes"].to_numpy()
base_level = df["PlayerLevel"].to_numpy()
base_ach = df["AchievementsUnlocked"].to_numpy()
base_purchases = df["InGamePurchases"].to_numpy()

# matches_last_7d: anchored on sessions, adjusted by engagement
matches_last_7d = clip_round(
    base_sessions * rng.uniform(0.9, 1.5, n) + engagement_score * 2.5 + rng.normal(0, 2.0, n),
    0,
    35,
)

# active_days_last_7d: closely tied to sessions/matches
active_days_last_7d = clip_round(
    np.minimum(7, np.maximum(0, matches_last_7d / rng.uniform(2.0, 4.2, n))) + engagement_score + rng.normal(0, 0.8, n),
    0,
    7,
)

# matches_last_30d: correlated with recent activity
matches_last_30d = clip_round(
    matches_last_7d * rng.uniform(2.5, 4.0, n) + rng.normal(0, 6.0, n),
    0,
    120,
)

# avg match duration in a mobile MOBA context
avg_match_duration_min = np.clip(base_avg_session / rng.uniform(4.8, 6.5, n) + rng.normal(0, 1.5, n), 8, 28)
avg_match_duration_min = np.round(avg_match_duration_min, 1)

# player_level re-anchored to a smaller mobile progression scale
player_level = clip_round(
    (base_level * 0.45) + (account_age_days / 10) + rng.normal(0, 4, n),
    1,
    60,
)

# tutorial completion
p_tutorial = np.where(account_age_days <= 7, 0.78, 0.97)
p_tutorial = np.where(player_level <= 5, p_tutorial - 0.12, p_tutorial)
p_tutorial = np.clip(p_tutorial, 0.55, 0.99)
tutorial_completion = (rng.random(n) < p_tutorial).astype(int)

# ranked match ratio
ranked_match_ratio = np.clip(
    sigmoid((player_level - 18) / 7) * rng.uniform(0.4, 1.0, n) + difficulty_score * 0.15 + rng.normal(0, 0.08, n),
    0,
    1,
)
ranked_match_ratio = np.round(ranked_match_ratio, 3)

# win rate last 10
win_rate_last_10 = np.clip(
    0.50 + (difficulty_score - 0.5) * 0.08 + rng.normal(0, 0.09, n),
    0.25,
    0.75,
)
win_rate_last_10 = np.round(win_rate_last_10, 3)

# party / premade behavior
party_rate = np.clip(
    0.25 + np.maximum(engagement_score, -0.5) * 0.18 + rng.beta(2, 3, n) * 0.45,
    0,
    1,
)
party_rate = np.round(party_rate, 3)

premade_ratio = np.clip(party_rate * rng.uniform(0.65, 1.15, n), 0, 1)
premade_ratio = np.round(premade_ratio, 3)

# frustration signals
report_count_30d = clip_round(
    rng.poisson(0.5 + (1 - np.clip(win_rate_last_10, 0.25, 0.75)) * 2.0 + (1 - party_rate) * 1.2 + difficulty_score * 0.8, n),
    0,
    8,
)

afk_count_30d = clip_round(
    rng.poisson(0.3 + (1 - party_rate) * 1.0 + report_count_30d * 0.15 + np.maximum(0, 0.4 - win_rate_last_10) * 2.0, n),
    0,
    6,
)

# last login gap: lower activity -> longer gap
last_login_gap_days = clip_round(
    8.5 - active_days_last_7d - matches_last_7d / 8 - engagement_score * 1.2 + afk_count_30d * 0.35 + rng.normal(0, 1.4, n),
    0,
    14,
)

# champion pool size and progression velocity
champion_pool_size = clip_round(
    3 + player_level / 3 + active_days_last_7d + rng.normal(0, 2.5, n),
    3,
    35,
)

progression_velocity_30d = np.clip(
    (matches_last_30d / 18) + (player_level / np.maximum(account_age_days, 5)) * 20 + rng.normal(0, 1.5, n),
    0,
    15,
)
progression_velocity_30d = np.round(progression_velocity_30d, 2)

# ranked unlock timing
rank_unlocked = (player_level >= 10).astype(int)
days_since_ranked_unlock = np.where(
    rank_unlocked == 1,
    np.clip(account_age_days - rng.integers(7, 35, n), 0, 330),
    -1,
)

# spend features
spender_flag = ((base_purchases > 0) | (rng.random(n) < (0.08 + matches_last_30d / 250))).astype(int)

spend_score = base_purchases + matches_last_30d / 30 + np.maximum(player_level - 15, 0) / 10
spend_tier = np.full(n, "none", dtype=object)
spend_tier[(spender_flag == 1) & (spend_score < 2.0)] = "low"
spend_tier[(spender_flag == 1) & (spend_score >= 2.0) & (spend_score < 5.0)] = "medium"
spend_tier[(spender_flag == 1) & (spend_score >= 5.0)] = "high"

days_since_last_purchase = np.where(
    spender_flag == 1,
    clip_round(30 - matches_last_7d / 3 - active_days_last_7d + rng.normal(15, 12, n), 0, 180),
    np.nan,
)

# -----------------------------
# 6. Rank tier assignment
# -----------------------------
rank_score = player_level * 0.45 + ranked_match_ratio * 28 + (win_rate_last_10 - 0.5) * 25 + progression_velocity_30d

rank_tier = np.full(n, "unranked", dtype=object)
rank_tier[(rank_score >= 12) & (rank_score < 18)] = "iron"
rank_tier[(rank_score >= 18) & (rank_score < 24)] = "bronze"
rank_tier[(rank_score >= 24) & (rank_score < 31)] = "silver"
rank_tier[(rank_score >= 31) & (rank_score < 38)] = "gold"
rank_tier[(rank_score >= 38) & (rank_score < 46)] = "platinum"
rank_tier[(rank_score >= 46) & (rank_score < 54)] = "emerald"
rank_tier[(rank_score >= 54) & (rank_score < 62)] = "diamond"
rank_tier[(rank_score >= 62)] = "master_plus"

# Keep some low-level users unranked
rank_tier[player_level < 10] = "unranked"
ranked_match_ratio[player_level < 10] = 0.0
days_since_ranked_unlock[player_level < 10] = -1

# -----------------------------
# 7. Synthetic churn-risk target
# -----------------------------
new_player_flag = (account_age_days <= 30).astype(int)
low_activity_flag = (matches_last_7d <= 3).astype(int)
low_active_days_flag = (active_days_last_7d <= 2).astype(int)
high_gap_flag = (last_login_gap_days >= 5).astype(int)
low_party_flag = (party_rate <= 0.20).astype(int)
low_progress_flag = (progression_velocity_30d <= 2.5).astype(int)
friction_flag = ((report_count_30d >= 3) | (afk_count_30d >= 2)).astype(int)
untutored_flag = (tutorial_completion == 0).astype(int)

risk_score = (
    1.8 * high_gap_flag
    + 1.6 * low_activity_flag
    + 1.3 * low_active_days_flag
    + 1.2 * untutored_flag * new_player_flag
    + 1.0 * low_party_flag
    + 0.9 * low_progress_flag
    + 0.8 * friction_flag
    + 0.5 * (spender_flag == 0).astype(int)
    + rng.normal(0, 0.45, n)
)

risk_prob = sigmoid(risk_score - 2.6)
threshold = np.quantile(risk_prob, 1 - TARGET_RATE)
churn_risk = (risk_prob >= threshold).astype(int)

# -----------------------------
# 8. Final synthetic WR dataset
# -----------------------------
out = pd.DataFrame({
    "player_id": np.arange(100001, 100001 + n),
    "region": region,
    "account_age_days": account_age_days,
    "device_tier": device_tier,
    "preferred_queue": preferred_queue,
    "preferred_role": preferred_role,
    "matches_last_7d": matches_last_7d,
    "matches_last_30d": matches_last_30d,
    "active_days_last_7d": active_days_last_7d,
    "avg_match_duration_min": avg_match_duration_min,
    "last_login_gap_days": last_login_gap_days,
    "player_level": player_level,
    "rank_tier": rank_tier,
    "ranked_match_ratio": np.round(ranked_match_ratio, 3),
    "win_rate_last_10": np.round(win_rate_last_10, 3),
    "party_rate": np.round(party_rate, 3),
    "premade_ratio": np.round(premade_ratio, 3),
    "tutorial_completion": tutorial_completion,
    "report_count_30d": report_count_30d,
    "afk_count_30d": afk_count_30d,
    "champion_pool_size": champion_pool_size,
    "days_since_ranked_unlock": days_since_ranked_unlock,
    "progression_velocity_30d": progression_velocity_30d,
    "spender_flag": spender_flag,
    "spend_tier": spend_tier,
    "days_since_last_purchase": days_since_last_purchase,
    "churn_risk": churn_risk,
})

# Optional cleanup for display consistency
out["days_since_last_purchase"] = out["days_since_last_purchase"].round(0)

# Save
out.to_csv(OUTPUT_PATH, index=False)

print(f"Saved: {OUTPUT_PATH}")
print(out.head())
print("\nChurn risk rate:")
print(out["churn_risk"].mean().round(4))
print("\nRegion distribution:")
print(out["region"].value_counts(normalize=True).round(3))