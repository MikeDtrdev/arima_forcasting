import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_realistic_crime_data():
    """Create realistic monthly crime data for Barangay 41 and 43"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create monthly date range from 2019 to 2024
    start_date = pd.Timestamp('2019-01-01')
    end_date = pd.Timestamp('2024-12-01')
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    barangays = ['Barangay 41', 'Barangay 43']
    crime_types = [
        'Theft', 'Assault', 'Vandalism', 'Fraud', 'Harassments', 
        'Breaking and Entering', 'Vehicle Theft', 'Drug-related', 'Domestic Violence'
    ]
    
    records = []
    
    for barangay in barangays:
        for crime_type in crime_types:
            # Set realistic base rates per crime type and barangay
            if barangay == 'Barangay 41':
                base_rates = {
                    'Theft': 8, 'Assault': 5, 'Vandalism': 4, 'Fraud': 3,
                    'Harassments': 4, 'Breaking and Entering': 2, 
                    'Vehicle Theft': 2, 'Drug-related': 3, 'Domestic Violence': 3
                }
            else:  # Barangay 43
                base_rates = {
                    'Theft': 6, 'Assault': 4, 'Vandalism': 3, 'Fraud': 2,
                    'Harassments': 3, 'Breaking and Entering': 1, 
                    'Vehicle Theft': 1, 'Drug-related': 2, 'Domestic Violence': 2
                }
            
            base_rate = base_rates[crime_type]
            
            # Generate monthly data with realistic patterns
            for i, date in enumerate(dates):
                # Add seasonal variation (higher in certain months)
                month = date.month
                seasonal_factor = 1.0
                if month in [12, 1, 2]:  # Holiday season
                    seasonal_factor = 1.3
                elif month in [6, 7, 8]:  # Summer
                    seasonal_factor = 1.2
                elif month in [3, 4, 5]:  # Spring
                    seasonal_factor = 0.9
                
                # Add slight upward trend over years
                year_factor = 1.0 + (i / len(dates)) * 0.2
                
                # Add some random variation
                random_factor = np.random.normal(1.0, 0.3)
                random_factor = max(0.1, random_factor)  # Ensure positive
                
                # Calculate final count
                monthly_count = int(base_rate * seasonal_factor * year_factor * random_factor)
                monthly_count = max(0, monthly_count)  # Ensure non-negative
                
                records.append({
                    'date': date,
                    'barangay': barangay,
                    'crime_type': crime_type,
                    'count': monthly_count
                })
    
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    print("Creating realistic crime data...")
    df = create_realistic_crime_data()
    
    # Save to CSV
    df.to_csv('data/realistic_crimes_2019_2024.csv', index=False)
    
    print(f"✅ Created {len(df)} records")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"🏢 Barangays: {df['barangay'].nunique()}")
    print(f"🔍 Crime types: {df['crime_type'].nunique()}")
    print(f"📊 Total incidents: {df['count'].sum()}")
    
    # Show sample data
    print("\n📋 Sample data:")
    print(df.head(10))
    
    # Show summary by barangay and crime type
    print("\n📈 Summary by Barangay and Crime Type:")
    summary = df.groupby(['barangay', 'crime_type'])['count'].agg(['mean', 'sum']).round(1)
    print(summary)
