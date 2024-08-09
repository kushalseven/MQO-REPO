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
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel

plans_df = pd.read_csv('testdata/multi_query_plans.csv')
savings_df = pd.read_csv('testdata/multi_query_savings.csv')

Q = {}

for _, row in plans_df.iterrows():
    plan_var = f'q_{row["multi_query_id"]}_{row["query_id"]}_{row["global_plan_id"]}'
    Q[(plan_var, plan_var)] = row["plan_cost"]

for _, row in savings_df.iterrows():
    plan_var1 = f'q_{row["multi_query_id"]}_{row["plan1_global_id"]}'
    plan_var2 = f'q_{row["multi_query_id"]}_{row["plan2_global_id"]}'
    Q[(plan_var1, plan_var2)] = row["cost_saving"]

for _, multi_query_group in plans_df.groupby('multi_query_id'):
    queries = multi_query_group.groupby('query_id')
    for _, query_group in queries:
        plans = query_group['global_plan_id'].tolist()
        for i in range(len(plans)):
            for j in range(i + 1, len(plans)):
                plan_var1 = f'q_{multi_query_group.iloc[0]["multi_query_id"]}_{query_group.iloc[0]["query_id"]}_{plans[i]}'
                plan_var2 = f'q_{multi_query_group.iloc[0]["multi_query_id"]}_{query_group.iloc[0]["query_id"]}_{plans[j]}'
                Q[(plan_var1, plan_var2)] = 2  # Penalty for selecting more than one plan per query

bqm = BinaryQuadraticModel.from_qubo(Q)

sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))
response = sampler.sample(bqm, num_reads=100)

best_solution = response.first.sample
print("Best solution:", best_solution)

# Calculate the total cost of the best solution
total_cost = response.first.energy
print("Total cost of the best solution:", total_cost)

# Extract the selected plans from the best solution
selected_plans = [key for key, value in best_solution.items() if value == 1]
print("Selected plans:", selected_plans)