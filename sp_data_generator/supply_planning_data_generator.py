import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import openai
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
import os


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

class SupplyPlanningDataGenerator:
    """Generate realistic supply planning data for testing optimization models."""
    
    def __init__(self, 
                 seed: Optional[int] = None,
                 openai_api_key: Optional[str] = None):
        """
        Initialize the data generator with optional seed for reproducibility.
        
        Args:
            seed: Random seed for reproducibility
            openai_api_key: API key for OpenAI services (if using GenAI features)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
    
    def generate_products(self, 
                         num_products: int = 5, 
                         category_distribution: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Generate product data with realistic attributes.
        
        Args:
            num_products: Number of products to generate
            category_distribution: Dictionary mapping categories to their probability
                                  e.g. {'Electronics': 0.3, 'Clothing': 0.4, 'Food': 0.3}
        
        Returns:
            DataFrame with product information
        """
        if category_distribution is None:
            category_distribution = {
                'Electronics': 0.3, 
                'Clothing': 0.4, 
                'Home Goods': 0.2, 
                'Food': 0.1
            }
        
        categories = list(category_distribution.keys())
        probabilities = list(category_distribution.values())
        
        data = {
            'product_id': [f'P{i:04d}' for i in range(1, num_products + 1)],
            'product_name': [f'Product {i}' for i in range(1, num_products + 1)],
            'category': np.random.choice(categories, size=num_products, p=probabilities),
            'unit_cost': np.round(np.random.uniform(5, 100, num_products), 2),
            'setup_cost': np.random.randint(100, 1000, num_products),
            'setup_time': np.random.randint(30, 240, num_products),
            'production_time': np.random.randint(5, 60, num_products),
            'min_production_qty': np.random.randint(10, 100, num_products),
            'shelf_life_days': np.random.choice([30, 60, 90, 180, 365, 730, 1095], num_products)
        }
        
        # Add realistic product weights
        data['weight_kg'] = np.round(np.random.exponential(5, num_products), 2)
        data['weight_kg'] = np.clip(data['weight_kg'], 0.1, 50)
        
        # Add realistic volumes
        data['volume_m3'] = np.round(data['weight_kg'] * np.random.uniform(0.001, 0.01, num_products), 4)
        
        return pd.DataFrame(data)
    
    def generate_facilities(self, 
                          num_facilities: int = 3,
                          facility_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate facility data including factories and warehouses.
        
        Args:
            num_facilities: Number of facilities to generate
            facility_types: List of facility types to choose from
        
        Returns:
            DataFrame with facility information
        """
        if facility_types is None:
            facility_types = ['Factory', 'Warehouse', 'Distribution Center']
            
        data = {
            'facility_id': [f'F{i:03d}' for i in range(1, num_facilities + 1)],
            'facility_name': [f'Facility {i}' for i in range(1, num_facilities + 1)],
            'facility_type': np.random.choice(facility_types, num_facilities),
            'latitude': np.random.uniform(30, 50, num_facilities),
            'longitude': np.random.uniform(-120, -70, num_facilities),
            'capacity': np.random.randint(5000, 20000, num_facilities),
            'operating_cost': np.random.randint(10000, 50000, num_facilities)
        }
        
        df = pd.DataFrame(data)
        
        # Add specific capacity for each facility type
        for i, row in df.iterrows():
            if row['facility_type'] == 'Factory':
                df.loc[i, 'production_capacity'] = np.random.randint(1000, 5000)
                df.loc[i, 'storage_capacity'] = np.random.randint(500, 2000)
            elif row['facility_type'] == 'Warehouse':
                df.loc[i, 'production_capacity'] = 0
                df.loc[i, 'storage_capacity'] = np.random.randint(5000, 15000)
            else:  # Distribution Center
                df.loc[i, 'production_capacity'] = np.random.randint(0, 1000)
                df.loc[i, 'storage_capacity'] = np.random.randint(2000, 8000)
                
        return df
    
    def generate_transportation_matrix(self, facilities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate transportation costs and times between facilities.
        
        Args:
            facilities_df: DataFrame with facility information
            
        Returns:
            DataFrame with transportation costs and times
        """
        facility_ids = facilities_df['facility_id'].tolist()
        n_facilities = len(facility_ids)
        
        data = []
        for i in range(n_facilities):
            for j in range(n_facilities):
                if i != j:
                    # Calculate distance based on lat/long (simplified)
                    lat1, lon1 = facilities_df.iloc[i][['latitude', 'longitude']]
                    lat2, lon2 = facilities_df.iloc[j][['latitude', 'longitude']]
                    
                    # Simple distance calculation (not accurate geographic distance)
                    distance = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111  # rough km conversion
                    
                    # Add some randomness for real-world variation in routes
                    distance = distance * np.random.uniform(0.8, 1.2)
                    
                    data.append({
                        'from_facility': facility_ids[i],
                        'to_facility': facility_ids[j],
                        'distance_km': round(distance, 2),
                        'transport_cost': round(distance * np.random.uniform(0.5, 2.0), 2),
                        'transport_time_hours': round(distance / np.random.uniform(40, 80), 2)
                    })
        
        return pd.DataFrame(data)
    
    def generate_demand_data(self, 
                           products_df: pd.DataFrame,
                           facilities_df: pd.DataFrame,
                           start_date: str = '2023-01-01',
                           periods: int = 52,
                           frequency: str = 'W',
                           seasonality: bool = True,
                           trend: bool = True,
                           noise_level: float = 0.2) -> pd.DataFrame:
        """
        Generate time series demand data with optional seasonality and trend.
        
        Args:
            products_df: DataFrame with product information
            facilities_df: DataFrame with facility information
            start_date: Starting date for time series
            periods: Number of periods to generate
            frequency: Frequency string (D=daily, W=weekly, M=monthly)
            seasonality: Whether to include seasonal patterns
            trend: Whether to include trend component
            noise_level: Level of random noise (0 to 1)
            
        Returns:
            DataFrame with demand data
        """
        product_ids = products_df['product_id'].tolist()
        warehouse_ids = facilities_df[facilities_df['facility_type'].isin(['Warehouse', 'Distribution Center'])]['facility_id'].tolist()
        
        date_range = pd.date_range(start=start_date, periods=periods, freq=frequency)
        
        data = []
        
        # Define base volumes by product category
        category_volumes = {
            'Electronics': np.random.randint(50, 200),
            'Clothing': np.random.randint(100, 500),
            'Home Goods': np.random.randint(30, 150),
            'Food': np.random.randint(200, 800)
        }
        
        for product_id in product_ids:
            product = products_df[products_df['product_id'] == product_id].iloc[0]
            category = product['category']
            base_volume = category_volumes.get(category, 100)
            
            # Generate demand for each facility
            for facility_id in warehouse_ids:
                # Create base demand with facility-specific scaling
                facility_scale = np.random.uniform(0.5, 1.5)
                base_demand = base_volume * facility_scale
                
                # Create time series components
                time_index = np.arange(len(date_range))
                
                # Trend component
                trend_component = np.zeros(len(time_index))
                if trend:
                    trend_slope = np.random.uniform(-0.01, 0.03)  # -1% to +3% change per period
                    trend_component = time_index * trend_slope * base_demand
                
                # Seasonal component
                seasonal_component = np.zeros(len(time_index))
                if seasonality:
                    if frequency == 'D':
                        # Daily seasonality (weekly pattern)
                        seasonal_component = 0.2 * base_demand * np.sin(2 * np.pi * time_index / 7)
                    elif frequency == 'W':
                        # Weekly seasonality (quarterly pattern)
                        seasonal_component = 0.3 * base_demand * np.sin(2 * np.pi * time_index / 13)
                    else:  # Monthly
                        # Monthly seasonality (yearly pattern)
                        seasonal_component = 0.4 * base_demand * np.sin(2 * np.pi * time_index / 12)
                
                # Noise component
                noise = np.random.normal(0, noise_level * base_demand, len(time_index))
                
                # Combine components and ensure non-negative
                demand = base_demand + trend_component + seasonal_component + noise
                demand = np.maximum(demand, 0).round()
                
                # Add to dataset
                for i, date in enumerate(date_range):
                    data.append({
                        'date': date,
                        'product_id': product_id,
                        'facility_id': facility_id,
                        'demand': int(demand[i])
                    })
        
        return pd.DataFrame(data)
    
    def generate_bill_of_materials(self, 
                                 products_df: pd.DataFrame, 
                                 max_components: int = 5,
                                 component_overlap: float = 0.3) -> pd.DataFrame:
        """
        Generate bill of materials (BOM) data for manufacturing.
        
        Args:
            products_df: DataFrame with product information
            max_components: Maximum number of components per product
            component_overlap: Probability of component reuse across products
            
        Returns:
            DataFrame with BOM data
        """
        
        # Generate components
        num_unique_components = int(len(products_df) * 2)  # 2x components as products
        components = [f'C{i:04d}' for i in range(1, num_unique_components + 1)]
        
        # Generate component properties
        component_data = {
            'component_id': components,
            'component_name': [f'Component {i}' for i in range(1, num_unique_components + 1)],
            'unit_cost': np.round(np.random.uniform(1, 50, num_unique_components), 2),
            'lead_time_days': np.random.randint(1, 30, num_unique_components)
        }
        components_df = pd.DataFrame(component_data)
        
        # Generate BOM relationships
        bom_data = []
        for _, product in products_df.iterrows():
            # Determine number of components for this product
            num_components = np.random.randint(2, max_components + 1)
            
            # Select components
            if np.random.random() < component_overlap and len(bom_data) > 0:
                # Reuse some components from existing BOMs
                existing_components = pd.DataFrame(bom_data)['component_id'].unique()
                if len(existing_components) > 0:
                    num_reused = min(np.random.randint(1, num_components), len(existing_components))
                    reused_components = np.random.choice(existing_components, num_reused, replace=False)
                    
                    for component_id in reused_components:
                        quantity = np.random.randint(1, 10)
                        bom_data.append({
                            'product_id': product['product_id'],
                            'component_id': component_id,
                            'quantity': quantity
                        })
                    
                    # Fill the rest with new components
                    remaining = num_components - num_reused
                    if remaining > 0:
                        new_components = np.random.choice(components, remaining, replace=False)
                        for component_id in new_components:
                            quantity = np.random.randint(1, 10)
                            bom_data.append({
                                'product_id': product['product_id'],
                                'component_id': component_id,
                                'quantity': quantity
                            })
            else:
                # Select all new components
                selected_components = np.random.choice(components, num_components, replace=False)
                for component_id in selected_components:
                    quantity = np.random.randint(1, 10)
                    bom_data.append({
                        'product_id': product['product_id'],
                        'component_id': component_id,
                        'quantity': quantity
                    })
        
        return pd.DataFrame(bom_data), components_df
    
    def generate_workforce_data(self, 
                              facilities_df: pd.DataFrame,
                              start_date: str = '2023-01-01',
                              periods: int = 52,
                              frequency: str = 'W') -> pd.DataFrame:
        """
        Generate workforce availability and cost data.
        
        Args:
            facilities_df: DataFrame with facility information
            start_date: Starting date for time series
            periods: Number of periods to generate
            frequency: Frequency string (D=daily, W=weekly, M=monthly)
            
        Returns:
            DataFrame with workforce data
        """
        factory_ids = facilities_df[facilities_df['facility_type'] == 'Factory']['facility_id'].tolist()
        date_range = pd.date_range(start=start_date, periods=periods, freq=frequency)
        
        data = []
        
        skill_levels = ['junior', 'intermediate', 'senior']
        
        for facility_id in factory_ids:
            # Base workforce size for this facility
            base_workforce = np.random.randint(20, 100)
            
            # Generate workforce fluctuations over time
            for date in date_range:
                # Add some seasonal and random variation
                day_of_year = date.dayofyear
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * day_of_year / 365)
                random_factor = np.random.uniform(0.9, 1.1)
                
                # Different workforce for different skill levels
                for skill in skill_levels:
                    if skill == 'junior':
                        skill_count = int(base_workforce * 0.5 * seasonal_factor * random_factor)
                        skill_cost = np.random.uniform(15, 25)
                    elif skill == 'intermediate':
                        skill_count = int(base_workforce * 0.3 * seasonal_factor * random_factor)
                        skill_cost = np.random.uniform(25, 40)
                    else:  # senior
                        skill_count = int(base_workforce * 0.2 * seasonal_factor * random_factor)
                        skill_cost = np.random.uniform(40, 60)
                    
                    data.append({
                        'date': date,
                        'facility_id': facility_id,
                        'skill_level': skill,
                        'available_workers': skill_count,
                        'hourly_cost': round(skill_cost, 2)
                    })
        
        return pd.DataFrame(data)
    
    def visualize_demand_patterns(self, demand_df: pd.DataFrame, product_ids: Optional[List[str]] = None):
        """
        Visualize demand patterns for selected products.
        
        Args:
            demand_df: DataFrame with demand data
            product_ids: List of product IDs to visualize (if None, select a random sample)
        """
        if product_ids is None:
            product_ids = random.sample(list(demand_df['product_id'].unique()), 3)
            
        plt.figure(figsize=(12, 8))
        
        for product_id in product_ids:
            product_demand = demand_df[demand_df['product_id'] == product_id]
            pivoted = product_demand.pivot_table(
                index='date', columns='facility_id', values='demand', aggfunc='sum'
            )
            total_demand = pivoted.sum(axis=1)
            plt.plot(total_demand.index, total_demand.values, label=f'Product {product_id}')
            
        plt.title('Demand Patterns by Product')
        plt.xlabel('Date')
        plt.ylabel('Total Demand')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def generate_realistic_product_names(self, categories: Dict[str, int]) -> List[str]:
        """
        Use GenAI to generate realistic product names based on categories.
        
        Args:
            categories: Dictionary mapping category names to number of products needed
            
        Returns:
            List of generated product names
        """
        if not self.openai_api_key:
            # Fallback to basic naming if no API key
            names = []
            for category, count in categories.items():
                for i in range(count):
                    names.append(f"{category} Product {i+1}")
            return names
        
        try:
            product_names = []
            
            for category, count in categories.items():
                prompt = f"Generate {count} realistic and specific product names for the category: {category}. Return just the names separated by commas, without any explanations."
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates realistic product names."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.7
                )
                
                # Extract names from response
                names_text = response.choices[0].message.content
                category_names = [name.strip() for name in names_text.split(',')][:count]
                product_names.extend(category_names)
                
            return product_names
            
        except Exception as e:
            print(f"Error generating product names with GenAI: {e}")
            # Fallback to basic naming
            names = []
            for category, count in categories.items():
                for i in range(count):
                    names.append(f"{category} Product {i+1}")
            return names
    
    def generate_full_dataset(self, 
                            num_products: int = 20,
                            num_facilities: int = 5,
                            periods: int = 52,
                            frequency: str = 'W',
                            start_date: str = '2023-01-01') -> Dict[str, pd.DataFrame]:
        """
        Generate a complete, integrated dataset for supply chain planning.
        
        Args:
            num_products: Number of products to generate
            num_facilities: Number of facilities to generate
            periods: Number of time periods
            frequency: Time frequency (D, W, M)
            start_date: Starting date for time series
            
        Returns:
            Dictionary of DataFrames containing the complete dataset
        """
        # Generate products
        products_df = self.generate_products(num_products)
        
        # Generate facilities
        facilities_df = self.generate_facilities(num_facilities)
        
        # Generate transportation matrix
        transport_df = self.generate_transportation_matrix(facilities_df)
        
        # Generate demand data
        demand_df = self.generate_demand_data(
            products_df=products_df,
            facilities_df=facilities_df,
            start_date=start_date,
            periods=periods,
            frequency=frequency
        )
        
        # Generate bill of materials
        bom_df, components_df = self.generate_bill_of_materials(products_df)
        
        # Generate workforce data
        workforce_df = self.generate_workforce_data(
            facilities_df=facilities_df,
            start_date=start_date,
            periods=periods,
            frequency=frequency
        )
        
        return {
            'products': products_df,
            'facilities': facilities_df,
            'transportation': transport_df,
            'demand': demand_df,
            'bill_of_materials': bom_df,
            'components': components_df,
            'workforce': workforce_df
        }
    
    def export_to_excel(self, dataset: Dict[str, pd.DataFrame], filename: str = 'supply_planning_data.xlsx'):
        """
        Export the generated dataset to Excel.
        
        Args:
            dataset: Dictionary of DataFrames to export
            filename: Excel filename
        """
        with pd.ExcelWriter(filename) as writer:
            for sheet_name, df in dataset.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

# Example usage
if __name__ == "__main__":
    generator = SupplyPlanningDataGenerator(seed=42)
    dataset = generator.generate_full_dataset()
    generator.export_to_excel(dataset)
    
    # Visualize some data
    generator.visualize_demand_patterns(dataset['demand'])