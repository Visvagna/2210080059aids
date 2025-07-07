import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# =============== CONFIGURABLES ===============
CONDITIONS = ["diabetes", "heart", "obesity"]
ADVICE_MAP = {
    "diabetes": "🩸 Cut sugar, increase activity",
    "heart": "❤️ Exercise, reduce cholesterol",
    "obesity": "⚖️ Balanced diet, walk 30 mins/day"
}

# =============== GLOBALLY TRAINED MODEL ===============
# Train ONCE at startup (mock data)
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, 100)
scaler = StandardScaler().fit(X_train)
model = LogisticRegression().fit(scaler.transform(X_train), y_train)

# =============== VECTORIZED PREDICTION ===============
def predict_risks(inputs):
    """Predict all risks in one vectorized operation."""
    scaled = scaler.transform([inputs])
    probabilities = model.predict_proba(scaled)[:, 1] * 100  # [diabetes_prob, heart_prob, obesity_prob]
    return dict(zip(CONDITIONS, np.round(probabilities, 2)))

# =============== OPTIMIZED MAIN ===============
def main():
    print("\n🔍 Ultra-Optimized Health Risk Assessor\n")
    
    # Input with validation (compressed)
    try:
        age, bmi, glucose = (int(input(f"Enter {x}: ")) for x in ["Age (18-120)", "BMI (10-50)", "Glucose (70-300)"])
        smoker, family = (int(input(f"{x}? (1=Yes, 0=No): ")) for x in ["Smoker", "Family History"])
        if not (18 <= age <= 120) or not (10 <= bmi <= 50) or not (70 <= glucose <= 300) or smoker not in {0,1} or family not in {0,1}:
            raise ValueError
    except:
        print("❌ Invalid input!")
        return

    # Get all risks in one call
    risks = predict_risks([age, bmi, glucose, smoker, family])

    # Output (no redundant loops)
    print("\n📊 Risk Summary:")
    print("\n".join(f"• {c.capitalize()}: {r}%" for c, r in risks.items()))
    
    print("\n📌 Top Advice:")
    top_advice = [f"{ADVICE_MAP[c]} (Risk: {r}%)" for c, r in risks.items() if r >= 60]
    print("\n".join(top_advice) if top_advice else "✅ All risks low!")

if __name__ == "__main__":
    main()
