import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/mnt/data/WeatherTech Championship Daytona Race.csv'
data = pd.read_csv(file_path)

# Convert Lap Time and Session Time to numerical format
def convert_time_to_seconds(time_str):
    try:
        minutes, seconds = map(float, time_str.split(':'))
        return minutes * 60 + seconds
    except ValueError:
        return None  # Handle rows with incorrect format

data['Lap Time (s)'] = data['Lap Time'].apply(convert_time_to_seconds)
data['Session Time (s)'] = data['Session Time'].apply(convert_time_to_seconds)

# Calculate stint lengths
data['Pit Stop'] = data['Session Time (s)'].diff().fillna(0) > 30  # Threshold of 30 seconds for pit stop
data['Stint Length (s)'] = 0

# Group data by Car to calculate stint lengths
for car, group in data.groupby('Car'):
    stint_length = 0
    for i in range(len(group)):
        if group.iloc[i]['Pit']:
            stint_length = 0
        else:
            stint_length += group.iloc[i]['Lap Time (s)']
        data.loc[group.index[i], 'Stint Length (s)'] = stint_length

# Calculate average lap times before and after pit stops
data['Prev Lap Time (s)'] = data['Lap Time (s)'].shift(1)
data['Next Lap Time (s)'] = data['Lap Time (s)'].shift(-1)
data['Avg Lap Time Before'] = data.groupby('Car')['Lap Time (s)'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
data['Avg Lap Time After'] = data.groupby('Car')['Next Lap Time (s)'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

# Calculate lap time degradation
data['Lap Time Degradation'] = data['Lap Time (s)'] - data['Prev Lap Time (s)']

# Define net time gain
data['Net Time Gain'] = data['Avg Lap Time After'] < data['Avg Lap Time Before']

# Drop rows with missing values
data = data.dropna(subset=['Avg Lap Time Before', 'Avg Lap Time After', 'Lap Time Degradation', 'Stint Length (s)', 'Net Time Gain'])

# Convert target to binary
data['Net Time Gain'] = data['Net Time Gain'].astype(int)

# Define features and target variable
features = ['Avg Lap Time Before', 'Avg Lap Time After', 'Lap Time Degradation', 'Stint Length (s)']
X = data[features]
y = data['Net Time Gain']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display the evaluation metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance
importance = model.coef_[0]
feature_importance = pd.Series(importance, index=features).sort_values(ascending=False)

# Visualize feature importance
plt.figure(figsize=(10, 7))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Predicting Net Time Gain')
plt.show()





import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/mnt/data/IWSC Rolex Race.csv'
data = pd.read_csv(file_path)

# Filter data for Green flag conditions and GTD class
gtd_data = data[(data['Flag'] == 'Green') & (data['Class'] == 'GTD')]

# Convert 'Lap Time' and 'Session Time' to timedelta
def convert_to_timedelta(time_str):
    minutes, seconds = time_str.split(':')
    minutes = int(minutes)
    seconds = float(seconds)
    return pd.Timedelta(minutes=minutes, seconds=seconds)

def convert_session_time_to_timedelta(time_str):
    parts = time_str.split(':')
    if len(parts) == 2:  # 'mm:ss.SSS' format
        return pd.Timedelta(minutes=int(parts[0]), seconds=float(parts[1]))
    elif len(parts) == 3:  # 'hh:mm:ss.SSS' format
        return pd.Timedelta(hours=int(parts[0]), minutes=int(parts[1]), seconds=float(parts[2]))
    else:
        raise ValueError("Unexpected time format")

# Correct the session time format
gtd_data['Session Time'] = gtd_data['Session Time'].apply(lambda x: '00:' + x if len(x.split(':')) == 2 else x)

# Apply the conversion functions
gtd_data['Lap Time'] = gtd_data['Lap Time'].apply(convert_to_timedelta)
gtd_data['Session Time'] = gtd_data['Session Time'].apply(convert_session_time_to_timedelta)

# Calculate average lap times for each car
average_lap_times_per_car = gtd_data.groupby('Car')['Lap Time'].mean().dt.total_seconds()

# Identify the two cars with the lowest average lap times
lowest_avg_cars = average_lap_times_per_car.nsmallest(2).index.tolist()

# Filter data for the two cars with the lowest average lap times
low_avg_cars_data = gtd_data[gtd_data['Car'].isin(lowest_avg_cars)]

# Identify stints by sequences of laps between pit stops and filter out stints with yellow flags
low_avg_cars_data['Stint'] = (low_avg_cars_data['Location'] == 'Pit').cumsum()
stint_data = low_avg_cars_data.groupby(['Car', 'Stint']).filter(lambda x: (x['Flag'] == 'Green').all())

# Define the new lap range for filtering
lap_range_new = (200, 232)

# Filter the data for the specified lap range
filtered_stint_data_new = stint_data[(stint_data['Lap'] >= lap_range_new[0]) & 
                                     (stint_data['Lap'] <= lap_range_new[1])]

# Function to remove anomalies using IQR method
def remove_anomalies(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Remove anomalies in lap times for the new filtered data
filtered_stint_data_new_no_anomalies = remove_anomalies(filtered_stint_data_new, 'Lap Time')

# Apply a rolling average to smooth the lap times
window_size = 10  # Window size for smoothing

# Plot the lap times for the two cars in each stint after removing anomalies, with red vertical lines indicating pit stops
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 12), sharex=True)
for i, car in enumerate(lowest_avg_cars):
    car_stint_data_filtered_new = filtered_stint_data_new_no_anomalies[filtered_stint_data_new_no_anomalies['Car'] == car]
    for stint in car_stint_data_filtered_new['Stint'].unique():
        stint_lap_times_filtered_new = car_stint_data_filtered_new[car_stint_data_filtered_new['Stint'] == stint]
        smoothed_lap_times_new = stint_lap_times_filtered_new['Lap Time'].dt.total_seconds().rolling(window=window_size).mean()
        axes[i].plot(stint_lap_times_filtered_new['Lap'], smoothed_lap_times_new, marker='o', linewidth=2, label=f'Stint {stint}')
        # Mark pit stops with red vertical lines
        pit_laps_new = car_stint_data_filtered_new[car_stint_data_filtered_new['Location'] == 'Pit']['Lap']
        for lap in pit_laps_new:
            axes[i].axvline(lap, color='red', linestyle='-', linewidth=2)  # Solid red vertical lines
    axes[i].set_title(f'Car {car} Lap Times Over Stints (No Anomalies, Laps {lap_range_new[0]}-{lap_range_new[1]})')
    axes[i].set_ylabel('Lap Time (seconds)')
    axes[i].legend()

axes[-1].set_xlabel('Lap Number')
plt.tight_layout()
plt.show()
