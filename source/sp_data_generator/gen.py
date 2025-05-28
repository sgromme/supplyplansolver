#!/usr/bin/env python3
# Basic usage



from supply_planning_data_generator import SupplyPlanningDataGenerator

# Initialize the generator (optionally with a seed for reproducibility)
generator = SupplyPlanningDataGenerator(seed=42)

# Generate a complete dataset
dataset = generator.generate_full_dataset(
    num_products=20,
    num_facilities=5,
    periods=52,
    frequency='W',  # Weekly data
    start_date='2023-01-01'
)

for key in dataset.keys():
    key_str = str(key) + ".csv"
    df = dataset[key]
    df.to_csv(key_str, index=True)
# Export to Excel for easy viewing or sharing
generator.export_to_excel(dataset, 'my_supply_chain_data.xlsx')