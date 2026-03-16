import numpy as np
from scipy import stats

# https://github.com/iorixxx/lucene-clueweb-retrieval/blob/master/src/main/java/edu/anadolu/knn/TStats.java
def TRisk(r, b, alpha):
    """
    Computes the T-Risk (Risk-reward tradeoff measure)

    Parameters:
    r (array-like): Effectiveness scores of the target system
    b (array-like): Effectiveness scores of the baseline system
    alpha (float): Risk aversion parameter (e.g., 1, 5, or 10).
                   Determines the magnitude of the penalty applied when the target
                   system underperforms compared to the baseline.

    Returns:
    tuple: (t_risk, p_value)
        - t_risk (float): The standardized risk-reward measure.
        - p_value (float): The 2-tailed p-value corresponding to the t-statistic.
    """
    r = np.array(r)
    b = np.array(b)

    if len(r) != len(b):
        raise ValueError("System and baseline score arrays must have the same length.")

    n = len(r)

    # 1. Calculate the per-query U_Risk differences
    # Formula: u_i = (r_i - b_i) - alpha * max(0, b_i - r_i)
    delta = r - b
    downside_penalty = alpha * np.maximum(0, b - r)
    u = delta - downside_penalty

    # 2. Calculate U_Risk (the sample mean of u)
    u_risk = np.mean(u)

    # 3. Calculate the sample standard deviation
    s = np.std(u, ddof=1)

    if s == 0:
        # Handle zero variance edge-case
        t_risk = np.inf if u_risk > 0 else (-np.inf if u_risk < 0 else 0.0)
        return t_risk, 1.0

    # 4. Calculate Standard Error of U_Risk
    se_u = s / np.sqrt(n)

    # 5. T-Risk is the studentized version of U_Risk
    t_risk = u_risk / se_u

    # 6. Calculate the statistical significance (p-value, degrees of freedom = n - 1)
    p_value = stats.t.sf(np.abs(t_risk), df=n - 1) * 2

    return t_risk, p_value

# --- Test ---
if __name__ == "__main__":
    ### https://github.com/iorixxx/lucene-clueweb-retrieval/blob/master/src/test/java/edu/anadolu/knn/TRiskTest.java

    # Example scores (e.g., Average Precision for 5 queries)
    base = [0.1021, 0.29326, 0.05711, 0.0, 0.05984, 0.0, 0.0, 0.0, 0.02862, 0.03518, 0.47569, 0.26307, 0.0, 0.0, 0.28504,
     0.03551, 0.06722, 0.32121, 0.0, 0.07245, 0.6107, 0.0, 0.0, 0.0, 0.22392, 0.0, 0.43172, 0.0, 0.06715, 0.28618,
     0.25482, 0.05495, 0.07206, 0.0, 0.01115, 0.0, 0.0, 0.13899, 0.2208, 0.10847, 0.0, 0.0, 0.0, 0.20123, 0.26957,
     0.27151, 0.01494, 0.13398, 0.0, 0.21972, 0.02392, 0.09307, 0.0, 0.0, 0.15874, 0.25934, 0.0548, 0.0, 0.06516,
     0.10489, 0.20117, 0.04759, 0.01289, 0.01809, 0.0, 0.23091, 0.02212, 0.0, 0.0, 0.0, 0.0, 0.38834, 0.01313, 0.04237,
     0.05625, 0.24475, 0.0, 0.00995, 0.41074, 0.01392, 0.0, 0.0, 0.53026, 0.21081, 0.83653, 0.08848, 0.04418, 0.00469,
     0.0, 0.07034, 0.0, 0.06682, 0.0, 0.09563, 0.01578, 0.0, 0.0, 0.48438, 0.55843, 0.06884, 0.14599, 0.2236, 0.02394,
     0.05397, 0.70238, 0.0, 0.0, 0.06305, 0.5, 0.0, 0.05843, 0.19402, 0.0, 0.00487, 0.14031, 0.03286, 0.0, 0.32829,
     0.31747, 0.28172, 0.13093, 0.0, 0.0, 0.28854, 0.09498, 0.65778, 0.50423, 0.0, 0.07996, 0.19394, 0.21306, 0.0, 0.0,
     0.03823, 0.5731, 0.0, 0.24298, 0.04607, 0.28139, 0.35779, 0.0, 0.03218, 0.24909, 0.26094, 0.5033, 0.07742, 0.0,
     0.31792, 0.0, 0.05777, 0.0, 0.0, 0.0, 0.0, 0.04738, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06117, 0.01056, 0.0, 0.0, 0.42797,
     0.0, 0.0, 0.0183, 0.03333, 0.01644, 0.00249, 0.0, 0.0, 0.0, 0.15626, 0.0, 0.01411, 0.02064, 0.0, 0.0, 0.00219, 0.0,
     0.0, 0.02476, 0.0, 0.0, 0.0534, 0.04037, 0.0, 0.14014, 0.10631, 0.0, 0.0, 0.28605, 0.12119, 0.03096, 0.01902]

    run = [0.11674, 0.50125, 0.39263, 0.03286, 0.09348, 0.01915, 0.0897, 0.0, 0.21404, 0.0, 0.4733, 0.11833, 0.0, 0.0,
     0.50526, 0.0, 0.34423, 0.36579, 0.0, 0.0, 0.64201, 0.0, 0.0506, 0.0, 0.26975, 0.1965, 0.01494, 0.02594, 0.29559,
     0.48415, 0.20119, 0.12181, 0.0, 0.0, 0.03736, 0.0, 0.03406, 0.08962, 0.32152, 0.24719, 0.0, 0.0, 0.07102, 0.09145,
     0.27752, 0.136, 0.05849, 0.27046, 0.2099, 0.191, 0.16424, 0.02001, 0.0, 0.0, 0.01184, 0.15889, 0.1314, 0.0,
     0.14151, 0.03717, 0.30188, 0.03308, 0.18457, 0.0, 0.0, 0.32468, 0.03152, 0.01567, 0.0, 0.0, 0.0, 0.21579, 0.0,
     0.12563, 0.26887, 0.50314, 0.16381, 0.02512, 0.53302, 0.02903, 0.0, 0.0, 0.67719, 0.12764, 0.85291, 0.05507,
     0.2059, 0.01793, 0.0078, 0.22178, 0.0, 0.27587, 0.0, 0.1392, 0.10458, 0.0163, 0.0, 0.61103, 0.38859, 0.33895,
     0.23036, 0.29874, 0.0, 0.06208, 0.69517, 0.0, 0.0, 0.18499, 0.0, 0.17441, 0.0773, 0.16251, 0.2517, 0.0, 0.23682,
     0.05898, 0.0, 0.25537, 0.69328, 0.21239, 0.30938, 0.13227, 0.0, 0.26223, 0.11983, 0.59691, 0.41932, 0.0, 0.03533,
     0.43117, 0.0, 0.0, 0.0, 0.0951, 0.34425, 0.07894, 0.06419, 0.04728, 0.22435, 0.14204, 0.02339, 0.2236, 0.34954,
     0.26293, 0.54413, 0.08238, 0.0, 0.02786, 0.0, 0.05777, 0.0, 0.41064, 0.0, 0.0, 0.07335, 0.23147, 0.0, 0.0, 0.0,
     0.00564, 0.0, 0.1928, 0.0, 0.0, 0.8603, 0.0, 0.0, 0.10936, 0.0058, 0.15103, 0.00337, 0.0, 0.0, 0.36012, 0.29187,
     0.0, 0.01566, 0.02308, 0.14214, 0.00481, 0.18835, 0.03732, 0.01075, 0.0, 0.03751, 0.0, 0.3101, 0.16474, 0.0,
     0.15298, 0.1467, 0.0, 0.03929, 0.13148, 0.0132, 0.01684, 0.23117]


    
    risk_score, p_val = TRisk(run, base,5) # -2.7131
    print(f"TRisk Score: {risk_score}")
    print(f"p_val Score: {p_val}")


    risk_score, p_val = TRisk(base, run,5) # -7.9998
    print(f"TRisk Score: {risk_score}")
    print(f"p_val Score: {p_val}")

    risk_score, p_val = TRisk(run, base,1) # 0.9003
    print(f"TRisk Score: {risk_score}")
    print(f"p_val Score: {p_val}")


    risk_score, p_val = TRisk(base, run,1) # -6.2345
    print(f"TRisk Score: {risk_score}")
    print(f"p_val Score: {p_val}")

    risk_score, p_val = TRisk(run, base,2) # -0.7165
    print(f"TRisk Score: {risk_score}")
    print(f"p_val Score: {p_val}")


    risk_score, p_val = TRisk(base, run,2) # -7.1108
    print(f"TRisk Score: {risk_score}")
    print(f"p_val Score: {p_val}")


# OUTPUTS
# TRisk Score: -2.7131225748098045
# p_val Score: 0.007258661416639113
# TRisk Score: -7.999758243395788
# p_val Score: 1.0654334805927146e-13
# TRisk Score: 0.9003475861994112
# p_val Score: 0.3690402072867014
# TRisk Score: -6.234494831656089
# p_val Score: 2.7266020737207978e-09
# TRisk Score: -0.7164676318930132
# p_val Score: 0.4745554664200783
# TRisk Score: -7.11081141940383
# p_val Score: 2.1048825619517632e-11