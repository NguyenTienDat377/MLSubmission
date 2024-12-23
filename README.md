# Machine Learning Project

# INT3405E 56

Team member :

- Nguyen Bao Long (msv)
- Nguyen Tien Dat (msv)
- Vu Tu Quynh (msv)

This is the version which gets the highest score, with references from public code sources with edits, of which the main source is : ⁦https://www.kaggle.com/code/cchangyyy/0-494-notebook⁩

## Key Features and Techniques

### 1. Handling Time-Series Data

def process\*file(filename, dirname):
df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet')) df.drop('step', axis=1, inplace=True) return df.describe().values.reshape(-1), filename.split('=')[1]

def load_time_series(dirname) -> pd.DataFrame:
ids = os.listdir(dirname)

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))

    stats, indexes = zip(*results)

    df = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
    df['id'] = indexes
    return df

_process_file:_

- Reads individual parquet files containing time-series data.
- Drops the unnecessary step column (as it does not contribute to predictions).
- Calculates and reshapes statistical summaries (mean, std, etc.).
- Extracts the unique id from the filename.

_load_time_series:_

- Reads all file IDs from a directory and processes them concurrently using ThreadPoolExecutor for faster computation.
- Combines the statistics into a DataFrame with each row corresponding to a participant.

### 2. Dimensionality Reduction

```
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim*3),
            nn.ReLU(),
            nn.Linear(encoding_dim*3, encoding_dim*2),
            nn.ReLU(),
            nn.Linear(encoding_dim*2, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, input_dim*3),
            nn.ReLU(),
            nn.Linear(input_dim*3, input_dim),
            nn.Sigmoid()
        )

Class AutoEncoder:
- Reduces high dimensional data into smaller latent features that retains important patterns



```

def perform_autoencoder(df, encoding_dim=50, epochs=50, batch_size=32):
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

    data_tensor = torch.FloatTensor(df_scaled)

    input_dim = data_tensor.shape[1]
    autoencoder = AutoEncoder(input_dim, encoding_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters())

    for epoch in range(epochs):
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i : i + batch_size]
            optimizer.zero_grad()
            reconstructed = autoencoder(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}]')

    with torch.no_grad():
        encoded_data = autoencoder.encoder(data_tensor).numpy()

    df_encoded = pd.DataFrame(encoded_data, columns=[f'Enc_{i + 1}' for i in range(encoded_data.shape[1])])

    return df_encoded

perform_autoencoder

- Use StandartScaler to normalize the input features
- The autoencoder model is created based on thr input dimensions
- After training using Mean-Squared Error, the data is passed through the encoder to compress latent data

def feature_engineering(df):
season_cols = [col for col in df.columns if 'Season' in col]
df = df.drop(season_cols, axis=1)
df['BMI_Age'] = df['Physical-BMI'] _ df['Basic_Demos-Age']
df['Internet_Hours_Age'] = df['PreInt_EduHx-computerinternet_hoursday'] _ df['Basic_Demos-Age']
df['BMI_Internet_Hours'] = df['Physical-BMI'] _ df['PreInt_EduHx-computerinternet_hoursday']
df['BFP_BMI'] = df['BIA-BIA_Fat'] / df['BIA-BIA_BMI']
df['FFMI_BFP'] = df['BIA-BIA_FFMI'] / df['BIA-BIA_Fat']
df['FMI_BFP'] = df['BIA-BIA_FMI'] / df['BIA-BIA_Fat']
df['LST_TBW'] = df['BIA-BIA_LST'] / df['BIA-BIA_TBW']
df['BFP_BMR'] = df['BIA-BIA_Fat'] _ df['BIA-BIA_BMR']
df['BFP_DEE'] = df['BIA-BIA_Fat'] _ df['BIA-BIA_DEE']
df['BMR_Weight'] = df['BIA-BIA_BMR'] / df['Physical-Weight']
df['DEE_Weight'] = df['BIA-BIA_DEE'] / df['Physical-Weight']
df['SMM_Height'] = df['BIA-BIA_SMM'] / df['Physical-Height']
df['Muscle_to_Fat'] = df['BIA-BIA_SMM'] / df['BIA-BIA_FMI']
df['Hydration_Status'] = df['BIA-BIA_TBW'] / df['Physical-Weight']
df['ICW_TBW'] = df['BIA-BIA_ICW'] / df['BIA-BIA_TBW']
df['BMI_PHR'] = df['Physical-BMI'] _ df['Physical-HeartRate']
df['SDS_InternetHours'] = df['SDS-SDS_Total_T'] \* df['PreInt_EduHx-computerinternet_hoursday']

    return df

feature_engineering

- Drop feature about seasons due to four seasons being similar
- Combining existing features, creating new features

train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
sample = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')

train_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet")
test_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet")

df_train = train_ts.drop('id', axis=1)
df_test = test_ts.drop('id', axis=1)

train_ts_encoded = perform_autoencoder(df_train, encoding_dim=60, epochs=100, batch_size=32)
test_ts_encoded = perform_autoencoder(df_test, encoding_dim=60, epochs=100, batch_size=32)

time_series_cols = train_ts_encoded.columns.tolist()
train_ts_encoded["id"]=train_ts["id"]
test_ts_encoded['id']=test_ts["id"]

train = pd.merge(train, train_ts_encoded, how="left", on='id')
test = pd.merge(test, test_ts_encoded, how="left", on='id')

- Read files from the inut
- Load time series from the series_train.parquet file
- Drop id features from input
- Perform autoencoder on the time series in order to reduce dimensionality
- Merge time series into training

imputer = KNNImputer(n_neighbors=5)
numeric_cols = train.select_dtypes(include=['int32', 'int64', 'float64', 'int64']).columns
imputed_data = imputer.fit_transform(train[numeric_cols])
train_imputed = pd.DataFrame(imputed_data, columns=numeric_cols)
train_imputed['sii'] = train_imputed['sii'].round().astype(int)
for col in train.columns:
if col not in numeric_cols:
train_imputed[col] = train[col]

train = train_imputed

train = feature_engineering(train)
train = train.dropna(thresh=10, axis=0)
test = feature_engineering(test)

- Handling missing data by using KNNImputer

featuresCols = ['Basic_Demos-Age', 'Basic_Demos-Sex',
'CGAS-CGAS_Score', 'Physical-BMI',
'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
'Fitness_Endurance-Max_Stage',
'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND',
'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU',
'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR',
'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone',
'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num',
'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
'BIA-BIA_TBW', 'PAQ_A-PAQ_A_Total',
'PAQ_C-PAQ_C_Total', 'SDS-SDS_Total_Raw',
'SDS-SDS_Total_T',
'PreInt_EduHx-computerinternet_hoursday', 'BMI_Age','Internet_Hours_Age','BMI_Internet_Hours',
'BFP_BMI', 'FFMI_BFP', 'FMI_BFP', 'LST_TBW', 'BFP_BMR', 'BFP_DEE', 'BMR_Weight', 'DEE_Weight',
'SMM_Height', 'Muscle_to_Fat', 'Hydration_Status', 'ICW_TBW', 'BMI_PHR', 'SDS_InternetHours']

featuresCols += time_series_cols
test = test[featuresCols]

- Add time series data into features data

def update(df):
global cat_c
for c in cat_c:
df[c] = df[c].fillna('Missing')
df[c] = df[c].astype('category')
return df

- Mark missing features and their respective category

train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
sample = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')

def process_file(filename, dirname):
df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
df.drop('step', axis=1, inplace=True)
return df.describe().values.reshape(-1), filename.split('=')[1]

def load_time_series(dirname) -> pd.DataFrame:
ids = os.listdir(dirname)

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))

    stats, indexes = zip(*results)

    df = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
    df['id'] = indexes
    return df


train_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet")
test_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet")

time_series_cols = train_ts.columns.tolist()
time_series_cols.remove("id")

train = pd.merge(train, train_ts, how="left", on='id')
test = pd.merge(test, test_ts, how="left", on='id')

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

featuresCols = ['Basic_Demos-Enroll_Season', 'Basic_Demos-Age', 'Basic_Demos-Sex',
'CGAS-Season', 'CGAS-CGAS_Score', 'Physical-Season', 'Physical-BMI',
'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
'Fitness_Endurance-Season', 'Fitness_Endurance-Max_Stage',
'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
'FGC-Season', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND',
'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU',
'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR',
'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone', 'BIA-Season',
'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num',
'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
'BIA-BIA_TBW', 'PAQ_A-Season', 'PAQ_A-PAQ_A_Total', 'PAQ_C-Season',
'PAQ_C-PAQ_C_Total', 'SDS-Season', 'SDS-SDS_Total_Raw',
'SDS-SDS_Total_T', 'PreInt_EduHx-Season',
'PreInt_EduHx-computerinternet_hoursday', 'sii']

featuresCols += time_series_cols

train = train[featuresCols]
train = train.dropna(subset='sii')

cat_c = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season',
'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season',
'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season']

def update(df):
global cat_c
for c in cat_c:
df[c] = df[c].fillna('Missing')
df[c] = df[c].astype('category')
return df

train = update(train)
test = update(test)

def create_mapping(column, dataset):
unique_values = dataset[column].unique()
return {value: idx for idx, value in enumerate(unique_values)}

for col in cat_c:
mapping = create_mapping(col, train)
mappingTe = create_mapping(col, test)

    train[col] = train[col].replace(mapping).astype(int)
    test[col] = test[col].replace(mappingTe).astype(int)

def quadratic_weighted_kappa(y_true, y_pred):
return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def threshold_Rounder(oof_non_rounded, thresholds):
return np.where(oof_non_rounded < thresholds[0], 0,
np.where(oof_non_rounded < thresholds[1], 1,
np.where(oof_non_rounded < thresholds[2], 2, 3)))

def evaluate_predictions(thresholds, y_true, oof_non_rounded):
rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
return -quadratic_weighted_kappa(y_true, rounded_p)

def TrainML(model_class, test_data):
X = train.drop(['sii'], axis=1)
y = train['sii']

    SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    train_S = []
    test_S = []

    oof_non_rounded = np.zeros(len(y), dtype=float)
    oof_rounded = np.zeros(len(y), dtype=int)
    test_preds = np.zeros((len(test_data), n_splits))

    for fold, (train_idx, test_idx) in enumerate(tqdm(SKF.split(X, y), desc="Training Folds", total=n_splits)):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        model = clone(model_class)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        oof_non_rounded[test_idx] = y_val_pred
        y_val_pred_rounded = y_val_pred.round(0).astype(int)
        oof_rounded[test_idx] = y_val_pred_rounded

        train_kappa = quadratic_weighted_kappa(y_train, y_train_pred.round(0).astype(int))
        val_kappa = quadratic_weighted_kappa(y_val, y_val_pred_rounded)

        train_S.append(train_kappa)
        test_S.append(val_kappa)

        test_preds[:, fold] = model.predict(test_data)

        print(f"Fold {fold+1} - Train QWK: {train_kappa:.4f}, Validation QWK: {val_kappa:.4f}")
        clear_output(wait=True)

    print(f"Mean Train QWK --> {np.mean(train_S):.4f}")
    print(f"Mean Validation QWK ---> {np.mean(test_S):.4f}")

    KappaOPtimizer = minimize(evaluate_predictions,
                              x0=[0.5, 1.5, 2.5], args=(y, oof_non_rounded),
                              method='Nelder-Mead')
    assert KappaOPtimizer.success, "Optimization did not converge."

    oof_tuned = threshold_Rounder(oof_non_rounded, KappaOPtimizer.x)
    tKappa = quadratic_weighted_kappa(y, oof_tuned)

    print(f"----> || Optimized QWK SCORE :: {Fore.CYAN}{Style.BRIGHT} {tKappa:.3f}{Style.RESET_ALL}")

    tpm = test_preds.mean(axis=1)
    tpTuned = threshold_Rounder(tpm, KappaOPtimizer.x)

    submission = pd.DataFrame({
        'id': sample['id'],
        'sii': tpTuned
    })
    return submission

Params = {
'learning_rate': 0.046,
'max_depth': 12,
'num_leaves': 478,
'min_data_in_leaf': 13,
'feature_fraction': 0.893,
'bagging_fraction': 0.784,
'bagging_freq': 4,
'lambda_l1': 10, # Increased from 6.59
'lambda_l2': 0.01, # Increased from 2.68e-06
'device': 'gpu'

}
Light = LGBMRegressor(\*\*Params, random_state=SEED, verbose=-1, n_estimators=300)
submission = TrainML(Light,test)

TrainML

- Split the data into features X and target Y
- Use Cross-validation K-FOld to slit the data to training and validation sets
- Trains the model on each fold and predicts on the training and validation sets
- Calculates the quadratic weighted kappa score for both set
- Stores out-of-fold predictions and predictions
- Optimizes thresholds for rounding predictions
- Apply optimized threshold to test predictions and prepare submission

- Define hyperparameter for the LGBM model with specific parameters
- Training the model using TrainML function with the LightBGM model and test data to generate predictions and create submissions
-
