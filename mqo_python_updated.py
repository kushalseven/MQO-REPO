import pandas as pd
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from dimod import BinaryQuadraticModel
import dwave.inspector
from logger_file import logger

# Initialize logger
logger.info(f"Logger initialized successfully in file")

# Load data
plans_df = pd.read_csv('data/multi_query_plans_100_7_2.csv')
savings_df = pd.read_csv('data/multi_query_savings_100_7_2.csv')
logger.info(f"Data Read Successfully!!!!!!!!!!")

# Initialize QUBO dictionary
Q = {}
logger.info(f"Initialized QUBO dictionary")

# Add the cost of each plan to the QUBO
for _, row in plans_df.iterrows():
    plan_var = f'q_{row["multi_query_id"]}_{row["query_id"]}_{row["global_plan_id"]}'
    Q[(plan_var, plan_var)] = row["plan_cost"]
logger.info(f"Added cost of each plan to QUBO matrix")

# Add the cost savings between plans to the QUBO
for _, row in savings_df.iterrows():
    plan_var1 = f'q_{row["multi_query_id"]}_{row["plan1_global_id"]}'
    plan_var2 = f'q_{row["multi_query_id"]}_{row["plan2_global_id"]}'
    if (plan_var1, plan_var2) not in Q:
        Q[(plan_var1, plan_var2)] = 0
    Q[(plan_var1, plan_var2)] += row["cost_saving"]
logger.info(f"Added the cost savings between plans to the QUBO matrix")

# Penalty for ensuring at least one plan is selected for each query in a multi-query
penalty_weight = 10
for multi_query_id, multi_query_group in plans_df.groupby('multi_query_id'):
    for query_id, query_group in multi_query_group.groupby('query_id'):
        plan_vars = [f'q_{multi_query_id}_{query_id}_{row["global_plan_id"]}'
                     for _, row in query_group.iterrows()]

        # Add a linear penalty term for the sum of selected plans
        for var in plan_vars:
            if (var, var) not in Q:
                Q[(var, var)] = 0
            Q[(var, var)] += penalty_weight
        
        # Add quadratic penalty to enforce that at least one plan is selected
        for i in range(len(plan_vars)):
            for j in range(i + 1, len(plan_vars)):
                if (plan_vars[i], plan_vars[j]) not in Q:
                    Q[(plan_vars[i], plan_vars[j])] = 0
                Q[(plan_vars[i], plan_vars[j])] += penalty_weight
logger.info(f"Penalized non-selection of plans within queries")

# Create BQM
bqm = BinaryQuadraticModel.from_qubo(Q)
logger.info(f"Created BQM")

# Initialize the sampler and sample the BQM
# sampler = EmbeddingComposite(DWaveSampler())
sampler = LeapHybridSampler()

logger.info(f"Sampler initialized")
try:
    # response = sampler.sample(bqm, num_reads = 10)
    response = sampler.sample(bqm)
except Exception as e:
    logger.error(f"Sampling failed with error: {e}")
logger.info(f"Sampled the data times")

# Extract the best solution
best_solution = response.first.sample
logger.info(f"Best Solution Achieved {best_solution}")

# Extract selected plans
selected_plans = [key for key, value in best_solution.items() if value == 1]
logger.info(f"Selected plans")

# Organize results by multi-query ID and query ID
results = []
for multi_query_id, multi_query_group in plans_df.groupby('multi_query_id'):
    for query_id, query_group in multi_query_group.groupby('query_id'):
        selected_plans_for_query = [
            plan for plan in selected_plans 
            if f'q_{multi_query_id}_{query_id}' in plan
        ]
        if selected_plans_for_query:
            results.append({
                'multi_query_id': multi_query_id,
                'query_id': query_id,
                'selected_plans': ','.join(selected_plans_for_query),
                'total_cost': response.first.energy  # Total cost is the same for each multi-query
            })

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv('output/selected_plans_results_0.9.csv', index=False)
logger.info(f"Results exported to 'selected_plans_results.csv'")

# Output the results
print("Results saved to 'selected_plans_results.csv'")
print(results_df)
dwave.inspector.show(response)