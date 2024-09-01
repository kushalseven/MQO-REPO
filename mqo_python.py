# Copyright [2024] [Kushal Bhattarai]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from dimod import BinaryQuadraticModel
from logger_file import logger

logger.info(f"Logger initialized successfully in file")

plans_df = pd.read_csv('data/multi_query_plans_7_2.csv')
savings_df = pd.read_csv('data/multi_query_savings_7_2.csv')
logger.info(f"Data Read Successfully!!!!!!!!!!")

Q = {}
logger.info(f"Initialized QUBO dictionary")

for _, row in plans_df.iterrows():
    plan_var = f'q_{row["multi_query_id"]}_{row["query_id"]}_{row["global_plan_id"]}'
    Q[(plan_var, plan_var)] = row["plan_cost"]
logger.info(f"Added cost of each plan to QUBO matrix")

for _, row in savings_df.iterrows():
    plan_var1 = f'q_{row["multi_query_id"]}_{row["plan1_global_id"]}'
    plan_var2 = f'q_{row["multi_query_id"]}_{row["plan2_global_id"]}'
    if (plan_var1, plan_var2) not in Q:
        Q[(plan_var1, plan_var2)] = 0
    Q[(plan_var1, plan_var2)] += row["cost_saving"]
logger.info(f"Added the cost savings between plans to the QUBO matrix")

# Added a penalty to enforce the selection of exactly K queries
K = 2  #number of queries to select
lambda_penalty = 10  # Penalty weight
logger.info(f"Added the penalty to enforce selection of exactly K queries")


# Create auxiliary variables to count selected queries per multi-query
for _, multi_query_group in plans_df.groupby('multi_query_id'):
    selected_queries = []
    for _, query_group in multi_query_group.groupby('query_id'):
        plan_vars = [f'q_{multi_query_group.iloc[0]["multi_query_id"]}_{query_group.iloc[0]["query_id"]}_{row["global_plan_id"]}'
                     for _, row in query_group.iterrows()]
        selected_queries.extend(plan_vars)

    # Add quadratic penalty term to enforce K selected queries
    for i in range(len(selected_queries)):
        for j in range(i + 1, len(selected_queries)):
            if (selected_queries[i], selected_queries[j]) not in Q:
                Q[(selected_queries[i], selected_queries[j])] = 0
            Q[(selected_queries[i], selected_queries[j])] += 2 * lambda_penalty


    # Add linear penalty term for deviation from K
    for var in selected_queries:
        if (var, var) not in Q:
            Q[(var, var)] = 0
        Q[(var, var)] += lambda_penalty * (1 - 2 * K)
logger.info(f"created auxillary variables to count selected queries per multi query")

bqm = BinaryQuadraticModel.from_qubo(Q)
logger.info(f"Created BQM")

# sampler = LeapHybridSampler()
sampler = EmbeddingComposite(DWaveSampler())
logger.info(f"sampler initialized")
try:
    response = sampler.sample(bqm)
except Exception as e:
    logger.error(f"Sampling failed with error: {e}")
logger.info(f"Sampled the data times")
best_solution = response.first.sample
print("Best solution:", best_solution)
logger.info(f"Best Solution Achieved{best_solution}")

# Calculate the total cost of the best solution
total_cost = response.first.energy
print("Total cost of the best solution:", total_cost)
logger.info(f"Total Cost{total_cost}")

# Extract the selected plans from the best solution
selected_plans = [key for key, value in best_solution.items() if value == 1]
print("Selected plans:", selected_plans)
logger.info(f"Selected plans")
 