
from  sp_data_generator.supply_planning_data_generator import SupplyPlanningDataGenerator as Sp

import os

generator = Sp(seed=42)
dataset = generator.generate_full_dataset()
generator.export_to_excel(dataset, filename='supply_planning_data.xlsx', working_directory=os.getcwd())
generator.export_to_parquet(dataset, working_directory=os.path.dirname(os.path.abspath(__file__)))

