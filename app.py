from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import os
import json
import io
import requests

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Khởi tạo Flask app
app = Flask(__name__, template_folder='templates')

class TemperaturePredictionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False  # Thêm dòng này
        self.feature_names = []

    def create_features(self, df):
        """Tạo các feature từ dữ liệu thời gian và vị trí"""
        df = df.copy()
        df['DATE'] = pd.to_datetime(df['DATE'])

        # Features từ thời gian
        df['year'] = df['DATE'].dt.year
        df['month'] = df['DATE'].dt.month
        df['day_of_year'] = df['DATE'].dt.dayofyear
        df['day_of_month'] = df['DATE'].dt.day

        # Features chu kỳ (sin/cos để capture tính tuần hoàn)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        # Features vị trí
        df['lat_lon_interaction'] = df['LATITUDE'] * df['LONGITUDE']
        df['distance_from_equator'] = np.abs(df['LATITUDE'])

        # Sắp xếp theo thời gian để tạo lag features
        df = df.sort_values(['LONGITUDE', 'LATITUDE', 'DATE'])
        
        df_dew_available = df.dropna(subset=['DEW']).copy()

        if not df_dew_available.empty:
            # Tính trung bình DEW theo Longitude, Latitude, và Day of Year
            # Sử dụng .reset_index() để chuyển groupby keys thành cột
            seasonal_dew_avg = df_dew_available.groupby(['LONGITUDE', 'LATITUDE', 'day_of_year'])['DEW'].mean().reset_index()
            seasonal_dew_avg.rename(columns={'DEW': 'seasonal_dew_avg'}, inplace=True)

            # 2. Merge giá trị DEW trung bình theo mùa vào DataFrame gốc
            # Merge dựa trên Longitude, Latitude, và Day of Year
            df = df.merge(seasonal_dew_avg, on=['LONGITUDE', 'LATITUDE', 'day_of_year'], how='left')

            # 3. Điền các giá trị NaN trong cột DEW gốc bằng giá trị seasonal_dew_avg
            # Chỉ điền cho những hàng mà DEW ban đầu là NaN
            df['DEW'] = df['DEW'].fillna(df['seasonal_dew_avg'])

            # Xóa cột trung gian seasonal_dew_avg
            df = df.drop(columns=['seasonal_dew_avg'])

        # Lag features (nhiệt độ ngày trước)
        df['temp_lag_1'] = df.groupby(['LONGITUDE', 'LATITUDE'])['TMP'].shift(1)
        df['temp_lag_7'] = df.groupby(['LONGITUDE', 'LATITUDE'])['TMP'].shift(7)
        df['temp_lag_30'] = df.groupby(['LONGITUDE', 'LATITUDE'])['TMP'].shift(30)

        # Moving averages - cách an toàn hơn
        df['temp_ma_7'] = df.groupby(['LONGITUDE', 'LATITUDE'])['TMP'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        df['temp_ma_30'] = df.groupby(['LONGITUDE', 'LATITUDE'])['TMP'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())

        df['DEW_lag_year'] = df.groupby(['LONGITUDE', 'LATITUDE'])['DEW'].shift(365) # Sử dụng DEW đã fillna theo mùa

        # Trend feature (số ngày từ ngày đầu tiên)
        df['days_from_start'] = (df['DATE'] - df['DATE'].min()).dt.days

        return df

    def prepare_data(self, df):
        """Chuẩn bị dữ liệu cho training"""
        df_features = self.create_features(df)

        # Chọn features cho model
        feature_cols = [
            'LONGITUDE', 'LATITUDE',
            'DEW', # <-- Sử dụng cột DEW đã được điền giá trị trung bình theo mùa
            'DEW_lag_year', # <-- Tùy chọn: vẫn giữ DEW_lag_year nếu muốn
            'cluster',
            'year', 'month', 'day_of_year', 'day_of_month',
            'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'lat_lon_interaction', 'distance_from_equator',
            'temp_lag_1', 'temp_lag_7', 'temp_lag_30',
            'temp_ma_7', 'temp_ma_30', 'days_from_start'
        ]

        self.feature_names = feature_cols

        # Loại bỏ rows có NaN (do lag features)
        df_clean = df_features.dropna(subset=self.feature_names + ['TMP'])

        X = df_clean[feature_cols]
        y = df_clean['TMP']

        return X, y, df_clean

    def train(self, df, test_size=0.2):
        """Training model"""
        print("Đang chuẩn bị dữ liệu...")
        X, y, df_clean = self.prepare_data(df)

        print(f"Số lượng samples sau khi làm sạch: {len(X)}")
        print(f"Features được sử dụng: {len(self.feature_names)}")

        # Chia train/test theo thời gian (test set là dữ liệu gần đây nhất)
        df_clean = df_clean.sort_values('DATE')
        split_idx = int(len(df_clean) * (1 - test_size))

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        print("Đang training model...")

        # Chuẩn hóa dữ liệu
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Training
        self.model.fit(X_train_scaled, y_train)

        # Đánh giá
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        print(f"\n=== KẾT QUẢ TRAINING ===")
        print(f"Train MAE: {train_mae:.3f}°C")
        print(f"Test MAE: {test_mae:.3f}°C")
        print(f"Train R²: {train_r2:.3f}")
        print(f"Test R²: {test_r2:.3f}")

        self.is_trained = True # Set True sau khi train

        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }

    def predict(self, df):
        """Dự đoán nhiệt độ"""
        if not self.is_trained:
            raise ValueError("Model chưa được training!")

        df_features = self.create_features(df)
        X = df_features[self.feature_names]

        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill')
        X = X.fillna(0)

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Chuyển đổi từ Kelvin sang Celsius
        predictions = predictions + 273.15
        
        # Giới hạn nhiệt độ trong khoảng hợp lý
        predictions = np.clip(predictions, -50, 50)

        return predictions

    def save_model(self, filepath):
        """Lưu model"""
        if not self.is_trained:
            raise ValueError("Model chưa được training!")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Model đã được lưu tại: {filepath}")

    def load_model(self, filepath):
        """Load model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained'] # Load lại trạng thái is_trained
        print(f"Model đã được load từ: {filepath}")

    def predict_single(self, longitude, latitude, date, dew, cluster):
        """Dự đoán cho một điểm dữ liệu đơn lẻ"""
        # Tạo DataFrame từ input
        df_single = pd.DataFrame({
            'DATE': [pd.to_datetime(date)],
            'LONGITUDE': [longitude],
            'LATITUDE': [latitude],
            'DEW': [dew],
            'cluster': [cluster],
            'TMP': [0]  # Dummy value
        })

        try:
            # Sử dụng phương thức predict chung, nó sẽ xử lý feature engineering và scaling
            prediction = self.predict(df_single)
            return prediction[0]
        except Exception as e:
            print(f"Lỗi khi dự đoán điểm đơn lẻ bằng model đã train: {e}")
            # Fallback: dự đoán đơn giản dựa trên DEW và vị trí (như ban đầu)
            print("Thực hiện dự đoán đơn giản làm fallback...")
            base_temp = dew + 5  # Rough estimate
            date_obj = pd.to_datetime(date)
            seasonal_factor = np.sin(2 * np.pi * date_obj.dayofyear / 365) * 10
            lat_factor = -abs(latitude * 0.5)
            return base_temp + seasonal_factor + lat_factor
def create_evaluator_from_trained_model(model, df, test_ratio=0.2):
    """Tạo evaluator từ model đã được train"""

    # Recreate the data processing
    df_features = model.create_features(df)
    feature_cols = model.feature_names # Lấy feature_names đã cập nhật từ model

    df_clean = df_features.dropna(subset=[f for f in feature_cols + ['TMP'] if f in df_features.columns])
    X = df_clean[feature_cols]
    y = df_clean['TMP']

    # Split data
    df_clean = df_clean.sort_values('DATE')
    split_idx = int(len(df_clean) * (1 - test_ratio))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Scale data
    X_train_scaled = model.scaler.transform(X_train) # Sử dụng scaler đã fit trong model
    X_test_scaled = model.scaler.transform(X_test)

    return df_clean.iloc[split_idx:]

# --- Cấu hình đường dẫn file ---
MODEL_PATH = 'temperature_model.pkl'
STATION_DATA_PATH = 'station_data.csv'
# --- THÊM: Đường dẫn đến file dữ liệu train gốc đã xử lý ---
# Đảm bảo file này tồn tại và có đủ các cột cần thiết
FULL_DATA_PATH = 'df_train.csv' # Hoặc '.parquet' nếu bạn lưu Parquet
# -----------------------------------------------------------

# Load model, dữ liệu trạm, VÀ dữ liệu train gốc khi app khởi động
model = None
station_coords = []
full_data_df = None # Biến để lưu DataFrame dữ liệu train gốc

try:
    # Load model
    model_data = joblib.load(MODEL_PATH)
    model_instance = TemperaturePredictionModel()
    model_instance.model = model_data['model']
    model_instance.scaler = model_data['scaler']
    model_instance.feature_names = model_data['feature_names']
    model_instance.is_trained = model_data['is_trained']
    model = model_instance
    print("Model loaded successfully.")

    # Load hoặc tạo danh sách tọa độ các trạm
    if os.path.exists(STATION_DATA_PATH):
        station_df = pd.read_csv(STATION_DATA_PATH)
        station_coords = station_df[['LONGITUDE', 'LATITUDE', 'cluster']].drop_duplicates().dropna(subset=['LONGITUDE', 'LATITUDE']).to_dict('records')
        print(f"Loaded {len(station_coords)} station coordinates.")

    # --- THÊM: Load dữ liệu train gốc ---
    if os.path.exists(FULL_DATA_PATH):
        print(f"Loading full data from {FULL_DATA_PATH}...")
        # Tùy thuộc vào định dạng file của bạn (.csv hoặc .parquet)
        if FULL_DATA_PATH.endswith('.csv'):
             full_data_df = pd.read_csv(FULL_DATA_PATH)
        elif FULL_DATA_PATH.endswith('.parquet'):
             full_data_df = pd.read_parquet(FULL_DATA_PATH)
        else:
             print("ERROR: Unsupported full data file format. Use .csv or .parquet.")
             full_data_df = None

        if full_data_df is not None:
             # Đảm bảo cột DATE là datetime và sắp xếp dữ liệu
             full_data_df['DATE'] = pd.to_datetime(full_data_df['DATE'])
             full_data_df = full_data_df.sort_values(['LONGITUDE', 'LATITUDE', 'DATE'])
             print(f"Full data loaded with {len(full_data_df)} rows.")
             print("Full data columns:", full_data_df.columns.tolist())

    else:
        print(f"ERROR: Full data file not found at {FULL_DATA_PATH}. Lag features will be less accurate or missing for early forecast dates.")
        full_data_df = None # Set về None nếu không tìm thấy file
    # -----------------------------------


except FileNotFoundError as e:
    print(f"Error loading file: {e}. Make sure files exist.")
    model = None
    station_coords = [{"LONGITUDE": 0, "LATITUDE": 0, "ERROR": "Data files not found"}]
    full_data_df = None
except Exception as e:
    print(f"An error occurred during startup: {e}")
    model = None
    station_coords = [{"LONGITUDE": 0, "LATITUDE": 0, "ERROR": "Startup failed"}]
    full_data_df = None


@app.route('/')
def index():
    """Trang chủ hiển thị form nhập liệu"""
    return render_template('index.html', stations=station_coords)

@app.route('/predict', methods=['POST'])
def predict():
    """API nhận request dự đoán"""
    if model is None or not model.is_trained:
         return jsonify({'error': 'Model not loaded or not trained'}), 500

    # THÊM: Kiểm tra dữ liệu train đã load chưa
    if full_data_df is None:
        return jsonify({'error': 'Dữ liệu train (df_train.csv) chưa sẵn sàng hoặc tải về bị lỗi. Vui lòng thử lại sau!'}), 500

    try:
        # Nếu là upload file CSV
        if 'csv_file' in request.files:
            file = request.files['csv_file']
            longitude = request.form.get('longitude', type=float)
            latitude = request.form.get('latitude', type=float)
            cluster = request.form.get('cluster', type=int)
            df_input = pd.read_csv(file)
            # Nếu thiếu các cột cần thiết thì báo lỗi
            required_cols = {'DATE', 'LONGITUDE', 'LATITUDE', 'DEW', 'cluster'}
            if not required_cols.issubset(df_input.columns):
                return jsonify({'error': f'CSV thiếu các cột: {required_cols - set(df_input.columns)}'}), 400
            # Nếu muốn chỉ dự đoán cho trạm đã chọn, lọc lại
            if longitude is not None and latitude is not None:
                df_input = df_input[
                    (df_input['LONGITUDE'] == longitude) &
                    (df_input['LATITUDE'] == latitude)
                ]
                if df_input.empty:
                    return jsonify({'error': 'File CSV không có dữ liệu cho trạm đã chọn'}), 400
            predictions = model.predict(df_input)
            df_input['TMP_PREDICTED'] = predictions
            df_input['DATE'] = pd.to_datetime(df_input['DATE']).dt.strftime('%Y-%m-%d')
            forecast_data = df_input[['DATE', 'TMP_PREDICTED']].to_dict('records')
            return jsonify({'forecast': forecast_data})

        # Nếu là random: chọn ngẫu nhiên 1 trạm từ danh sách trạm
        if request.is_json:
            data = request.get_json()
            if data.get('random'):
                if not station_coords:
                    return jsonify({'error': 'Không có danh sách trạm để random'}), 400
                import random
                random_station = random.choice(station_coords)
                longitude = random_station['LONGITUDE']
                latitude = random_station['LATITUDE']
                cluster = int(random_station['cluster'])
                # Lấy ngày bắt đầu từ dữ liệu train
                start_date_forecast = full_data_df['DATE'].max()
                forecast_days = 24
                end_date_forecast = start_date_forecast + pd.Timedelta(days=forecast_days)
                future_dates = pd.date_range(start=start_date_forecast, end=end_date_forecast, freq='D')[1:forecast_days+1]
                forecast_data_list = []
                for date in future_dates:
                    forecast_data_list.append({
                        'DATE': date,
                        'LONGITUDE': longitude,
                        'LATITUDE': latitude,
                        'DEW': np.nan,
                        'cluster': cluster,
                        'TMP': np.nan
                    })
                df_forecast_future = pd.DataFrame(forecast_data_list)
                df_forecast_future['DATE'] = pd.to_datetime(df_forecast_future['DATE'])
                # Lấy lịch sử cho trạm này
                days_needed_for_lags = 366
                history_start_date = start_date_forecast - pd.Timedelta(days=days_needed_for_lags)
                df_history_for_station = full_data_df[
                    (full_data_df['LONGITUDE'] == longitude) &
                    (full_data_df['LATITUDE'] == latitude) &
                    (full_data_df['DATE'] >= history_start_date) &
                    (full_data_df['DATE'] < start_date_forecast)
                ].copy()
                df_combined_for_forecast = pd.concat([df_history_for_station, df_forecast_future], ignore_index=True)
                df_combined_for_forecast['DATE'] = pd.to_datetime(df_combined_for_forecast['DATE'])
                df_combined_for_forecast = df_combined_for_forecast.sort_values('DATE').reset_index(drop=True)
                predictions = model.predict(df_combined_for_forecast)
                df_combined_for_forecast['TMP_PREDICTED'] = predictions
                df_forecast_results = df_combined_for_forecast[
                    df_combined_for_forecast['DATE'].isin(future_dates)
                ].copy()
                df_forecast_results['DATE'] = df_forecast_results['DATE'].dt.strftime('%Y-%m-%d')
                forecast_data = df_forecast_results[['DATE', 'TMP_PREDICTED']].to_dict('records')
                return jsonify({'forecast': forecast_data})

        # Mặc định: dự đoán theo trạm như cũ
        data = request.json
        longitude = float(data['longitude'])
        latitude = float(data['latitude'])
        # end_date_str = data['endDate']  # Không dùng nữa

        # Lấy ngày bắt đầu từ dữ liệu train
        start_date_forecast = full_data_df['DATE'].max()
        # end_date_forecast = pd.to_datetime(end_date_str)  # Không dùng nữa

        # Luôn dự đoán 24 ngày tiếp theo
        forecast_days = 24
        end_date_forecast = start_date_forecast + pd.Timedelta(days=forecast_days)

        # 1. Tìm cluster cho trạm đã chọn
        cluster = 0 # Giá trị mặc định
        selected_station = next((item for item in station_coords if item.get("LONGITUDE") == longitude and item.get("LATITUDE") == latitude), None)
        if selected_station and 'cluster' in selected_station:
             cluster = int(selected_station['cluster'])
        else:
             print(f"WARNING: Cluster not found for station {longitude}, {latitude} in loaded station_data. Using default cluster 0.")

        # 2. Tạo DataFrame cho các ngày tương lai
        future_dates = pd.date_range(start=start_date_forecast, end=end_date_forecast, freq='D')
        # Chỉ lấy 24 ngày tiếp theo (không lấy ngày start_date_forecast nếu muốn loại bỏ ngày cuối cùng của train)
        future_dates = future_dates[1:forecast_days+1]
        forecast_data_list = []
        for date in future_dates:
             forecast_data_list.append({
                'DATE': date,
                'LONGITUDE': longitude,
                'LATITUDE': latitude,
                'DEW': np.nan, # Sẽ được fillna sau khi kết hợp với lịch sử và tạo features
                'cluster': cluster,
                'TMP': np.nan # Giá trị cần dự đoán
             })
        df_forecast_future = pd.DataFrame(forecast_data_list)
        df_forecast_future['DATE'] = pd.to_datetime(df_forecast_future['DATE'])

        # 3. Lấy dữ liệu lịch sử cần thiết cho trạm này
        df_history_for_station = pd.DataFrame() # DataFrame rỗng mặc định
        if full_data_df is not None:
            # Xác định ngày bắt đầu lịch sử cần thiết
            # Cần đủ ngày để tính lag features xa nhất (temp_lag_30, DEW_lag_year)
            # DEW_lag_year cần 365 ngày trước start_date_forecast
            days_needed_for_lags = 366 # Thêm 1 ngày an toàn
            history_start_date = start_date_forecast - pd.Timedelta(days=days_needed_for_lags)

            # Lọc dữ liệu lịch sử cho trạm và khoảng ngày cần thiết
            df_history_for_station = full_data_df[
                (full_data_df['LONGITUDE'] == longitude) &
                (full_data_df['LATITUDE'] == latitude) &
                (full_data_df['DATE'] >= history_start_date) &
                (full_data_df['DATE'] < start_date_forecast) # Lấy đến trước ngày bắt đầu dự báo
            ].copy()
            print(f"Loaded {len(df_history_for_station)} history rows for forecast.")
            print("History data head:")
            print(df_history_for_station.head())
            print("History data tail:")
            print(df_history_for_station.tail())

        else:
             print("WARNING: Full data not loaded, cannot include history for lag features.")

        # 4. Kết hợp dữ liệu lịch sử và dữ liệu dự báo tương lai
        df_combined_for_forecast = pd.concat([df_history_for_station, df_forecast_future], ignore_index=True)
        df_combined_for_forecast['DATE'] = pd.to_datetime(df_combined_for_forecast['DATE'])
        df_combined_for_forecast = df_combined_for_forecast.sort_values('DATE').reset_index(drop=True)

        print("\nCombined DataFrame head:")
        print(df_combined_for_forecast.head())
        print("Combined DataFrame tail:")
        print(df_combined_for_forecast.tail())
        print(f"Combined DataFrame size: {len(df_combined_for_forecast)}")

        # 5. Sử dụng mô hình để dự đoán trên DataFrame kết hợp
        predictions = model.predict(df_combined_for_forecast)

        # Debug logging
        print("\nPrediction stats before post-processing:")
        print(f"Min temp: {predictions.min():.2f}°C")
        print(f"Max temp: {predictions.max():.2f}°C")
        print(f"Mean temp: {predictions.mean():.2f}°C")

        # Thêm dự đoán vào DataFrame
        df_combined_for_forecast['TMP_PREDICTED'] = predictions

        # 6. Trích xuất chỉ phần dự báo tương lai từ kết quả cuối cùng
        df_forecast_results = df_combined_for_forecast[
            df_combined_for_forecast['DATE'].isin(future_dates)
        ].copy()

        # 7. Chuẩn bị dữ liệu trả về JSON
        df_forecast_results['DATE'] = df_forecast_results['DATE'].dt.strftime('%Y-%m-%d')
        forecast_data = df_forecast_results[['DATE', 'TMP_PREDICTED']].to_dict('records')

        return jsonify({'forecast': forecast_data})

    except ValueError as ve:
        return jsonify({'error': f'Invalid input: {ve}'}), 400
    except Exception as e:
        print(f"Prediction error: {e}")
        # In traceback đầy đủ ra console server
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An error occurred during prediction: {e}'}), 500

@app.route('/mobile')
def mobile():
    """Trang mobile hiển thị form nhập liệu"""
    return render_template('mobile.html', stations=station_coords)

# Khi chạy file này trực tiếp bằng python app.py (không khuyến khích cho uvicorn),
# nó sẽ chạy asgi_app với uvicorn.

# --- Tải file df_train.csv nếu chưa có ---
DF_TRAIN_URL = "https://www.dropbox.com/scl/fi/txhte8yp8zs2fjoe21vp1/df_train.csv?rlkey=6ze6tcylq6vkpj0irhcaknkz8&st=9xzic2cf&dl=1"
DF_TRAIN_PATH = "df_train.csv"

def download_if_not_exists():
    if not os.path.exists(DF_TRAIN_PATH):
        print("Downloading large CSV file from Dropbox...")
        r = requests.get(DF_TRAIN_URL, stream=True)
        if r.status_code != 200:
            print("Download failed! Status code:", r.status_code)
            return False
        with open(DF_TRAIN_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
    return True

download_success = download_if_not_exists()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
