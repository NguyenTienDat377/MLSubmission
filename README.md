# Machine Learning Project

# INT3405E 56

Team member :

- Nguyen Bao Long (22028276)
- Nguyen Tien Dat (22028043)
- Vu Tu Quynh (22028253)

This is the version which gets the highest score, with references from public code sources with edits, of which the main source is : ⁦https://www.kaggle.com/code/cchangyyy/0-494-notebook⁩

## Key Features and Techniques

### 1. Handling Time-Series Data
```
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
```
**process_file:**

- Reads individual parquet files containing time-series data.
- Drops the unnecessary step column (as it does not contribute to predictions).
- Calculates and reshapes statistical summaries (mean, std, etc.).
- Extracts the unique id from the filename.

**load_time_series:**

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
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

```
**AutoEncoder:**
- Implements a neural network with an encoder to compress input data into lower dimensions and a decoder to reconstruct it.
- Uses ReLU activation in the hidden layers and Sigmoid for the final layer.




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
```
**perform_autoencoder :**

- Use StandartScaler to normalize the input features
- The autoencoder model is created based on thr input dimensions
- After training using Mean-Squared Error, the data is passed through the encoder to compress latent data


### 3. Feature Engineering
```
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
```

**feature_engineering:**

- Combines features to create new one
- Deriving new features introduces domain knowledge into the dataset, which can highlight important relationships not directly captured by raw features.
- Adds more predictive power to the dataset by introducing meaningful feature interactions.


### 4. Handling Missing Data

```
imputer = KNNImputer(n_neighbors=5)
numeric_cols = train.select_dtypes(include=['int32', 'int64', 'float64', 'int64']).columns
imputed_data = imputer.fit_transform(train[numeric_cols])
```
- Uses KNNImputer to fill missing values by considering the nearest neighbors.
- Ensures data completeness while preserving relationships between features.
- Processes only numeric columns and maintains non-numeric columns unchanged.


### 5. Threshold Rounder
```
def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1,
                             np.where(oof_non_rounded < thresholds[2], 2, 3)))
```
- Converts continuous predictions into discrete classes based on a set of thresholds :
    + Values below thresholds[0] are classified as 0.
    + Values between thresholds[0] and thresholds[1] are classified as 1, and so on.
- Ensures the output aligns with discrete target labels.

### 6. Evaluate Prediction
```
def evaluate_predictions(thresholds, y_true, oof_non_rounded):
    rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_p)
```
- Rounds the non-rounded predictions (oof_non_rounded) into discrete classes using the given thresholds.
- Calculates the negative QWK score for the rounded predictions compared to the true labels.

### 7. TrainML function
```
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

        ...
```
- Use StratifiedKFold to split the data into folds, maintaining label ratio.
- Save evaluation scores (QWK) on each fold.
- Save predictions (both continuous and discrete) on the training and test sets.
- Loop performs training and testing of the model for each fold in the Stratified K-Fold Cross- Validation process.
- After the cross-validation loop ends :
  + Calculate the average QWK score over all folds
  + Optimize classification threshold Apply optimal thresholds to both training (out-of-fold predictions) and test data.
  + Return the result

### 8. Train and make predictions
```
imputer = SimpleImputer(strategy='median')

ensemble = VotingRegressor(estimators=[
    ('lgb', Pipeline(steps=[('imputer', imputer), ('regressor', LGBMRegressor(random_state=SEED))])),
    ('xgb', Pipeline(steps=[('imputer', imputer), ('regressor', XGBRegressor(random_state=SEED))])),
    ('cat', Pipeline(steps=[('imputer', imputer), ('regressor', CatBoostRegressor(random_state=SEED, silent=True))])),
    ('rf', Pipeline(steps=[('imputer', imputer), ('regressor', RandomForestRegressor(random_state=SEED))])),
    ('gb', Pipeline(steps=[('imputer', imputer), ('regressor', GradientBoostingRegressor(random_state=SEED))]))
])
...
```
- Missing values in the data will be replaced with the median value of the corresponding column. 
- Create ensemble of 5 regression models (LightGBM, XGBoost, CatBoost, Random Forest, Gradient Boosting) 
- Use the TrainML function to train the ensemble on the training set and make predictions for the testing set. 
- Save prediction results as CSV file.

## Final Result
| **ID**        | **SII** |
|----------------|---------|
| 00008ff9      | 2       |
| 000fd460      | 0       |
| 00105258      | 0       |
| 00115b9f      | 0       |
| 0016bb22      | 1       |
| 001f3379      | 1       |
| 0038ba98      | 0       |
| 0068a485      | 0       |
| 0069fbed      | 2       |
| 0083e397      | 0       |
| 0087dd65      | 1       |
| 00abe655      | 0       |
| 00ae59c9      | 2       |
| 00af6387      | 1       |
| 00bd4359      | 2       |
| 00c0cd71      | 2       |
| 00d56d4b      | 0       |
| 00d9913d      | 0       |
| 00e6167c      | 0       |
| 00ebc35d      | 1       |
