#!/usr/bin/env python
import pandas as pd
import pulp

def optimizer_arguments_helper():
    optimization_args_dict = {}
    df_demand = pd.read_csv("../../demand.csv")
    df_facilities = pd.read_csv("../../facilities.csv")
    df_products = pd.read_csv("../../products.csv")
    tmp = {}
    product_cost = []
    for _, row in df_products.iterrows():
        product_cost.append(row["product_cost"])
    for _, row in df_facilities.iterrows():
        tmp[row["facility_id"]] = row["throughput_capacity"]
    for _, row in df_demand.iterrows():
        key_value = row['date'] + '-' + row['facility_id']
        if key_value not in optimization_args_dict:
            optimization_args_dict[key_value] = ([row['demand']], row['facility_id'] ,tmp[row['facility_id']])
        else:
            optimization_args_dict[key_value][0].append(row['demand'])
    return optimization_args_dict, product_cost

# Class that holds methods for optimiziing weekly production plans
class WeeklyProductionOptimizer:
    
    def __init__(self):
        pass
    
    def linear_programming_optimizer(self):
        # Still using the same W = 1500 right here as the data generator has not been fixed to produce meaningful capacity values ranging from 500 to 2000
        #   MAKE SURE TO FIX THIS TO ACTUALLY USE THE CAPACITY VALUES FROM THE FACILITIES DATA 
        optimization_args_dict, product_cost = optimizer_arguments_helper()
        W = 1500
        solved_demands_lp = {}
        for date in optimization_args_dict:
            demands = optimization_args_dict[date][0]
            cap = optimization_args_dict[date][2]
            values = [di * vi for di, vi in zip(demands, product_cost)]  
            model_new = pulp.LpProblem("Knapsack", pulp.LpMaximize)
            x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(demands))]
            model_new += pulp.lpSum(values[i]*x[i] for i in range(len(demands)))
            model_new += pulp.lpSum(demands[i]*x[i] for i in range(len(demands))) <= W #cap will be here in future
            model_new.solve(pulp.PULP_CBC_CMD(msg=False))
            chosen = [i for i in range(len(demands)) if pulp.value(x[i]) == 1]
            best_value = pulp.value(model_new.objective)
            total_weight = sum(demands[i] for i in chosen)
            solved_demands_lp[date] = (chosen, round(best_value, 3), total_weight, optimization_args_dict[date][1], cap)
        columns = ["Week and Facility", "Chosen Products", "Max Value", "Total Weight", "Facility ID", "Capacity"]
        solved_demands_lp_df = pd.DataFrame(columns=columns)
        rows = []
        for date in solved_demands_lp:
            rows.append([date] + list(solved_demands_lp[date]))
        solved_demands_lp_df = pd.DataFrame(rows, columns=columns)
        solved_demands_lp_df.to_csv("optimized_production_plans_lp.csv", index=False)
        print("///////////////////////////////////")
        print("LP Optimization completed. Results saved to 'optimized_production_plans_lp.csv'.")
        
    def dynamic_programming_optimizer(self):
        # Still using the same W = 1500 right here as the data generator has not been fixed to produce meaningful capacity values ranging from 500 to 2000
        #   MAKE SURE TO FIX THIS TO ACTUALLY USE THE CAPACITY VALUES FROM THE FACILITIES DATA 
        optimization_args_dict, product_cost = optimizer_arguments_helper()
        W = 1500
        solved_demands_dp = {}
        for date in optimization_args_dict:
            p = optimization_args_dict[date][0]
            cap = optimization_args_dict[date][2]
            #W = optimization_args_dict[date][2]
            Val = [pt*vt for pt, vt in zip(p, product_cost)]
            dp_table = [[0] * (W + 1) for _ in range(len(p)+ 1)]
            retain = [[False] * (W + 1) for _ in range(len(p)+ 1)] 
            for i in range(1, len(p) + 1):
                    p_i = p[i - 1]
                    v_i = Val[i - 1]
                    for curr_cap in range(W + 1):
                        best_value = dp_table[i - 1][curr_cap]
                        if p_i <= curr_cap:
                            candidate_value = dp_table[i - 1][curr_cap - p_i] + v_i
                            if candidate_value > best_value:
                                best_value = candidate_value
                                retain[i][curr_cap] = True
                        dp_table[i][curr_cap] = best_value         
            chosen = []
            temp_w = W
            n = len(p)
            for i in range(n, 0, -1):
                if retain[i][temp_w]:
                    chosen.append(i-1)
                    temp_w -= p[i-1]
                    if temp_w == 0:
                        break
            chosen.reverse()
            best_value = dp_table[n][W]
            total_weight = sum([p[i] for i in chosen])
            solved_demands_dp[date] = (chosen, round(best_value, 3), total_weight, optimization_args_dict[date][1], cap)
        columns = ["Week and Facility", "Chosen Products", "Max Value", "Total Weight", "Facility ID", "Capacity"]
        solved_demands_dp_df = pd.DataFrame(columns=columns)
        rows = []
        for date in solved_demands_dp:
            rows.append([date] + list(solved_demands_dp[date]))
        solved_demands_dp_df = pd.DataFrame(rows, columns=columns)
        solved_demands_dp_df.to_csv("optimized_production_plans_dp.csv", index=False)
        print("///////////////////////////////////")
        print("DP Optimization completed. Results saved to 'optimized_production_plans_dp.csv'.")
            
        
def main():
    optimizer = WeeklyProductionOptimizer()
    optimizer.dynamic_programming_optimizer()
            
if __name__ == "__main__":
    main()
    