# Farmer Profitability & Price Risk Simulator
# Author: Your Name
# Description: Python-based crop profitability and risk analysis using Monte Carlo simulation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Crop Data
# -----------------------------
crop_data = {
    "Crop": ["Rice", "Wheat", "Maize", "Cotton"],
    "Seed_Cost": [1200, 1000, 900, 1500],
    "Fertilizer_Cost": [2500, 2200, 2000, 3500],
    "Labour_Cost": [6000, 5000, 4500, 8000],
    "Irrigation_Cost": [3000, 2500, 2000, 4000],
    "Pesticide_Cost": [1800, 1500, 1200, 2500],
    "Yield_Quintal": [25, 22, 30, 18]
}

df = pd.DataFrame(crop_data)

# -----------------------------
# 2. Price History (₹ per quintal)
# -----------------------------
price_history = {
    "Rice": [1900, 2100, 2200, 2000, 2300],
    "Wheat": [2200, 2400, 2350, 2500, 2450],
    "Maize": [1600, 1700, 1800, 1750, 1650],
    "Cotton": [6000, 6500, 6200, 6800, 7000]
}

# -----------------------------
# 3. Total Cost Calculation
# -----------------------------
df["Total_Cost"] = (
    df["Seed_Cost"]
    + df["Fertilizer_Cost"]
    + df["Labour_Cost"]
    + df["Irrigation_Cost"]
    + df["Pesticide_Cost"]
)

# -----------------------------
# 4. Monte Carlo Simulation
# -----------------------------
SIMULATIONS = 1000
results = []

for _, row in df.iterrows():
    prices = price_history[row["Crop"]]
    avg_price = np.mean(prices)
    price_risk = np.std(prices)

    simulated_prices = np.random.normal(avg_price, price_risk, SIMULATIONS)
    simulated_income = simulated_prices * row["Yield_Quintal"]
    simulated_profit = simulated_income - row["Total_Cost"]

    results.append({
        "Crop": row["Crop"],
        "Avg_Profit": round(simulated_profit.mean(), 2),
        "Worst_Profit": round(np.percentile(simulated_profit, 5), 2),
        "Best_Profit": round(np.percentile(simulated_profit, 95), 2),
        "Loss_Probability_%": round((simulated_profit < 0).mean() * 100, 2)
    })

result_df = pd.DataFrame(results)

# -----------------------------
# 5. Risk-Based Decision
# -----------------------------
def decision(loss_prob):
    if loss_prob < 10:
        return "Highly Recommended"
    elif loss_prob < 30:
        return "Moderate Risk"
    else:
        return "Not Recommended"

result_df["Decision"] = result_df["Loss_Probability_%"].apply(decision)

# -----------------------------
# 6. Print Results
# -----------------------------
print("\n==============================")
print(" Crop Profitability & Risk Summary ")
print("==============================\n")
print(result_df)

# -----------------------------
# 7. Plot Profit Comparison
# -----------------------------
plt.figure()
plt.bar(result_df["Crop"], result_df["Avg_Profit"])
plt.title("Average Profit by Crop (Monte Carlo Simulation)")
plt.xlabel("Crop")
plt.ylabel("Profit (₹)")
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Final Recommendation
# -----------------------------
best_crop = result_df.sort_values("Loss_Probability_%").iloc[0]

print("\n==============================")
print(" FINAL CROP RECOMMENDATION ")
print("==============================")
print("Crop:", best_crop["Crop"])
print("Average Profit: ₹", int(best_crop["Avg_Profit"]))
print("Loss Probability:", best_crop["Loss_Probability_%"], "%")
print("Decision:", best_crop["Decision"])
